import os
import warnings

# ── Offline patch ──────────────────────────────────────────────────────────────
def bypass_zenodo_check(*args, **kwargs):
    base_dir = "/media/data_dump/Mann/rushil_temp/jetnet_data"
    arg_str  = str(args) + str(kwargs)
    if "g150"  in arg_str or "gluon" in arg_str or "'g'" in arg_str:
        return os.path.join(base_dir, "g150.hdf5")
    elif "q150" in arg_str or "quark" in arg_str or "'q'" in arg_str:
        return os.path.join(base_dir, "q150.hdf5")
    elif "t150" in arg_str or "top"   in arg_str or "'t'" in arg_str:
        return os.path.join(base_dir, "t150.hdf5")
    for a in args:
        if isinstance(a, str) and a.endswith(".hdf5"):
            return a
    return os.path.join(base_dir, "g150.hdf5")

import jetnet.datasets.utils, jetnet.datasets.jetnet
jetnet.datasets.utils.checkDownloadZenodoDataset  = bypass_zenodo_check
jetnet.datasets.jetnet.checkDownloadZenodoDataset = bypass_zenodo_check
print("Monkeypatch applied.")
# ──────────────────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
from jetnet.datasets import JetNet
from sklearn.neighbors import kneighbors_graph
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import ChebConv
from torch_geometric.utils import to_dense_batch
from scipy.stats import wasserstein_distance
import math

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
SAVE_DIR   = "/media/data_dump/Mann/rushil_temp/results_v4"
os.makedirs(SAVE_DIR, exist_ok=True)
LOG_FILE   = os.path.join(SAVE_DIR, "log.txt")

with open(LOG_FILE, "w") as f:
    f.write("=== Pipeline V4: DistancePool + DetBottleneck + ClusterDecoder + FiLM + PhysicsReg ===\n\n")

def log_print(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_print(f"Device: {device}")
if torch.cuda.is_available():
    log_print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE          = 2048
BATCH_SIZE_FLOW     = 2048
MAX_JETS            = 300000
GNN_HIDDEN_DIM      = 256
GNN_K               = 6
KNN_K               = 10
TOTAL_PARTICLES     = 150
NUM_CLUSTERS        = 10
PARTICLES_PER_CLUSTER = TOTAL_PARTICLES // NUM_CLUSTERS   # 15
GNN_EMB_DIM         = NUM_CLUSTERS * GNN_HIDDEN_DIM       # 2560

BOTTLENECK_DIM      = 512          # deterministic 2560 → 512
DECODER_HIDDEN      = 512          # hidden dim inside each cluster MLP
DIFFUSION_HIDDEN    = 2048
EPOCHS_AE           = 500
EPOCHS_DIFF         = 1000

NUM_JET_TYPES       = 3            # g=0  q=1  t=2
JET_TYPE_EMBED_DIM  = 8
COND_DIM_TOTAL      = 3 + JET_TYPE_EMBED_DIM   # [pt, mass, mult, type_emb] = 11

CHECKPOINT_EVERY    = 50
CHECKPOINT_DIR      = os.path.join(SAVE_DIR, "checkpoints")
WEIGHTS_DIR         = os.path.join(SAVE_DIR, "weights")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR,    exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _delete_old_numbered_ckpts(prefix):
    for fname in os.listdir(CHECKPOINT_DIR):
        if (fname.startswith(f"{prefix}_epoch_")
                and "latest" not in fname
                and fname.endswith(".pth")):
            try:
                os.remove(os.path.join(CHECKPOINT_DIR, fname))
                log_print(f"  Deleted old checkpoint: {fname}")
            except OSError:
                pass


def save_checkpoint(data_dict, prefix, epoch):
    numbered = os.path.join(CHECKPOINT_DIR, f"{prefix}_epoch_{epoch}.pth")
    latest   = os.path.join(CHECKPOINT_DIR, f"{prefix}_epoch_latest.pth")
    torch.save(data_dict, numbered)
    torch.save(data_dict, latest)
    _delete_old_numbered_ckpts(prefix)
    log_print(f"  [{prefix}] Checkpoint saved at epoch {epoch}")


def load_latest_checkpoint(prefix):
    latest = os.path.join(CHECKPOINT_DIR, f"{prefix}_epoch_latest.pth")
    if not os.path.exists(latest):
        return None
    ckpt = torch.load(latest, map_location=device)
    log_print(f"  [{prefix}] Resumed from epoch {ckpt['epoch']}")
    return ckpt


def cleanup_checkpoints(prefix):
    for fname in os.listdir(CHECKPOINT_DIR):
        if fname.startswith(f"{prefix}_epoch_") and fname.endswith(".pth"):
            try:
                os.remove(os.path.join(CHECKPOINT_DIR, fname))
            except OSError:
                pass
    log_print(f"  [{prefix}] All checkpoints deleted.")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_jetnet_data():
    DATA_DIR = "/media/data_dump/Mann/rushil_temp/jetnet_data"
    log_print(f"Loading JetNet from {DATA_DIR}")
    all_p, all_j, all_t = [], [], []
    for tid, jtype in enumerate(['g', 'q', 't']):
        p, j = JetNet.getData(
            jet_type=[jtype], data_dir=DATA_DIR,
            particle_features=['etarel', 'phirel', 'ptrel', 'mask'],
            jet_features=['pt', 'eta', 'mass', 'num_particles'],
            num_particles=TOTAL_PARTICLES, split='train', download=False
        )
        all_p.append(p)
        all_j.append(j)
        all_t.append(np.full(len(p), tid, dtype=np.int64))
        log_print(f"  {jtype}: {len(p)} jets")
    return (np.concatenate(all_p), np.concatenate(all_j),
            np.concatenate(all_t))


def collect_graph_and_targets(particle_data, jet_data, jet_types, max_jets=MAX_JETS):
    graph_data, target_particles, original_jets, collected_types = [], [], [], []
    failed = 0
    n = min(len(particle_data), max_jets)
    log_print(f"Building graphs for {n} jets...")

    for i in range(n):
        try:
            jp   = particle_data[i]
            mask = jp[:, 3] == 1
            vp   = jp[mask]
            if len(vp) == 0:
                failed += 1; continue
            vp   = vp[np.argsort(-vp[:, 2])]
            hits = vp[:, :3]

            k = KNN_K if len(hits) > KNN_K else len(hits) - 1
            if k <= 0:
                failed += 1; continue
            adj = kneighbors_graph(hits[:, :2], n_neighbors=k,
                                   mode='connectivity', include_self=False)
            edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
            graph_data.append(Data(x=torch.tensor(hits, dtype=torch.float),
                                   edge_index=edge_index))

            padded = np.zeros((TOTAL_PARTICLES, 4))
            padded[:len(vp)] = vp
            padded[:len(vp), 3] = 1
            target_particles.append(torch.tensor(padded, dtype=torch.float))
            original_jets.append((jp, jet_data[i]))
            collected_types.append(jet_types[i])

            if i % 50000 == 0 and i > 0:
                print(f"  {i}/{n} processed")
        except Exception:
            failed += 1

    log_print(f"Collected {len(graph_data)}, failed {failed}")
    return (graph_data, target_particles, original_jets,
            np.array(collected_types, dtype=np.int64))


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1: DISTANCE-POOL GNN ENCODER
# ══════════════════════════════════════════════════════════════════════════════

class DistancePoolChebNet(nn.Module):
    """
    4-layer ChebConv GNN with learnable soft cluster pooling in (η,φ) space.
    Each particle is soft-assigned to a cluster via softmax(-T · dist²),
    removing the pT-sort → fixed-slice ordering bias of the original encoder.
    Output: [B, NUM_CLUSTERS × GNN_HIDDEN_DIM] = [B, 2560]
    """
    def __init__(self, in_dim=3, hidden=GNN_HIDDEN_DIM, K=GNN_K,
                 n_clusters=NUM_CLUSTERS):
        super().__init__()
        self.conv1 = ChebConv(in_dim, hidden, K)
        self.conv2 = ChebConv(hidden,  hidden, K)
        self.conv3 = ChebConv(hidden,  hidden, K)
        self.conv4 = ChebConv(hidden,  hidden, K)
        self.bn1   = nn.BatchNorm1d(hidden)
        self.bn2   = nn.BatchNorm1d(hidden)
        self.bn3   = nn.BatchNorm1d(hidden)
        self.hidden     = hidden
        self.n_clusters = n_clusters
        self.centers    = nn.Parameter(torch.randn(n_clusters, 2) * 0.2)
        self.log_temp   = nn.Parameter(torch.tensor(2.0))

    def forward(self, x, edge_index, batch=None):
        h = F.leaky_relu(self.bn1(self.conv1(x, edge_index)))
        h = F.leaky_relu(self.bn2(self.conv2(h, edge_index)))
        h = F.leaky_relu(self.bn3(self.conv3(h, edge_index)))
        h = F.normalize(self.conv4(h, edge_index), p=2, dim=1)

        dense_h, node_mask = to_dense_batch(h,       batch, max_num_nodes=TOTAL_PARTICLES)
        dense_c, _         = to_dense_batch(x[:, :2], batch, max_num_nodes=TOTAL_PARTICLES)

        dist       = ((dense_c.unsqueeze(2) - self.centers) ** 2).sum(-1)
        T          = torch.exp(self.log_temp.clamp(-2.0, 3.0))
        assignment = F.softmax(-T * dist, dim=-1) * node_mask.unsqueeze(-1).float()

        pooled = torch.einsum('bnk,bnd->bkd', assignment, dense_h)
        pooled = pooled / (assignment.sum(1, keepdim=True).transpose(1, 2) + 1e-8)
        return pooled.reshape(pooled.size(0), -1)   # [B, 2560]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2: DETERMINISTIC BOTTLENECK
#
# Replaces VAE. No KL, no reparameterisation.
# 2560 → 512 via MLP encoder.  Trained end-to-end with GNN and ClusterDecoder
# so the decoder adapts to compressed 512-dim representations from day one.
# Z-score normalisation applied after encoding — clean because there is no
# probabilistic semantic being broken (unlike with a VAE + z-score).
# ══════════════════════════════════════════════════════════════════════════════

class DeterministicBottleneck(nn.Module):
    def __init__(self, in_dim=GNN_EMB_DIM, z_dim=BOTTLENECK_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.GELU(),
            nn.Linear(1024,   512),  nn.GELU(),
            nn.Linear(512,  z_dim),
        )
        # decoder kept for completeness but unused at inference
        self.decoder = nn.Sequential(
            nn.Linear(z_dim,  512),  nn.GELU(),
            nn.Linear(512,   1024),  nn.GELU(),
            nn.Linear(1024, in_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3: CLUSTER DECODER
#
# Architecture:
#   512 → MLP decompressor → 2560 → reshape [B, 10, 256] → shared_net → particles
#
# The MLP decompressor (512→1024→2560) reconstructs the structured cluster
# embedding space that the GNN encoder originally produced. This is trained
# jointly end-to-end so it learns a reconstruction driven by particle-level
# loss, not just MSE on raw embeddings (which is what a separate VAE decoder
# was doing). A single linear 512→2560 would be a rank-512 approximation with
# no ability to learn nonlinear manifold structure — the MLP with GELU fixes this.
#
# shared_net processes all clusters in parallel via batch reshape (faster and
# better generalisation than 10 independent MLPs), with BatchNorm + Dropout
# matching the original working decoder exactly.
# ══════════════════════════════════════════════════════════════════════════════

class ClusterDecoder(nn.Module):
    """
    Takes 512-dim bottleneck latent → reconstructs 150 particles.

    Step 1 — MLP decompressor: 512 → 1024 → 2560
        Learned nonlinear expansion back to the GNN cluster embedding space.
        Trained jointly so it optimises for particle reconstruction, not
        just embedding MSE.

    Step 2 — Cluster reshape: 2560 → [B, 10, 256]
        Natural split matching the GNN's distance-pooling cluster structure.

    Step 3 — Shared net: processes all 10 clusters simultaneously via
        view(-1, 256), shared weights across clusters for better generalisation.
        Each cluster independently decodes 15 particles → total 150 particles.
    """
    def __init__(self, latent_dim=BOTTLENECK_DIM,
                 cluster_emb_dim=GNN_HIDDEN_DIM,
                 particles_per_cluster=PARTICLES_PER_CLUSTER,
                 particle_dim=4):
        super().__init__()
        self.cluster_emb_dim       = cluster_emb_dim       # 256
        self.particles_per_cluster = particles_per_cluster  # 15
        self.particle_dim          = particle_dim           # 4

        # MLP decompressor: 512 → 1024 → 2560
        # Nonlinear so it can learn structured cluster geometry, not just
        # a rank-512 linear approximation of the 2560-dim space.
        self.decompressor = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, NUM_CLUSTERS * cluster_emb_dim),  # 2560
            # No final activation — GNN outputs are L2-normalised (freely pos/neg).
            # GELU here clips large negatives toward zero, preventing reconstruction
            # of the full signed embedding range.
        )

        # Shared MLP across all clusters (same weights per cluster)
        # Applied via reshape: [B×10, 256] → [B×10, 15×4]
        # BatchNorm + Dropout match original working decoder
        self.shared_net = nn.Sequential(
            nn.Linear(cluster_emb_dim, DECODER_HIDDEN),
            nn.BatchNorm1d(DECODER_HIDDEN),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(DECODER_HIDDEN, DECODER_HIDDEN),
            nn.BatchNorm1d(DECODER_HIDDEN),
            nn.LeakyReLU(0.2),
            nn.Linear(DECODER_HIDDEN, particles_per_cluster * particle_dim),
        )

    def forward(self, z):
        """
        z: [B, 512]  bottleneck latent
        returns: [B, 150, 4]  (eta, phi, pt, mask)
        """
        B = z.size(0)

        # Step 1: decompress 512 → 2560
        z_exp = self.decompressor(z)                              # [B, 2560]

        # Step 2: split into cluster embeddings
        z_clusters = z_exp.view(B, NUM_CLUSTERS, self.cluster_emb_dim)  # [B, 10, 256]

        # Step 3: shared net over all clusters simultaneously
        z_merged        = z_clusters.view(-1, self.cluster_emb_dim)     # [B×10, 256]
        particles_flat  = self.shared_net(z_merged)                     # [B×10, 60]
        particles_clust = particles_flat.view(B, NUM_CLUSTERS,
                                              self.particles_per_cluster,
                                              self.particle_dim)         # [B, 10, 15, 4]
        particles_full  = particles_clust.view(B, -1, self.particle_dim) # [B, 150, 4]

        return self._apply_activations(particles_full)

    def _apply_activations(self, output):
        """
        Physics-aware activations matching JetNet data format:
          etarel / phirel : tanh × 0.8  → [-0.8, 0.8]
          ptrel           : softmax × mask / sum  → sum≈1.0 per jet
          mask            : sigmoid × 2.0  → soft binary
        """
        eta  = torch.tanh(output[:, :, 0]) * 0.8
        phi  = torch.tanh(output[:, :, 1]) * 0.8
        mask = torch.sigmoid(output[:, :, 3] * 2.0)

        # pT: softmax ensures positivity + global sum=1, mask zeros invalid slots,
        # renormalise so valid particles sum exactly to 1.0
        pt_raw = F.softmax(output[:, :, 2], dim=1)
        pt     = pt_raw * mask
        pt     = pt / (pt.sum(dim=1, keepdim=True) + 1e-8)

        return torch.stack([eta, phi, pt, mask], dim=2)


# ══════════════════════════════════════════════════════════════════════════════
# LOSSES
# ══════════════════════════════════════════════════════════════════════════════

def masked_wasserstein_loss(pred, target):
    """
    Sorted-L1 proxy Wasserstein on η, φ, pT independently.
    Matches the original working AE loss exactly.
    """
    pred_mask   = pred[:, :, 3]
    target_mask = (target[:, :, 3] > 0.5).float()

    pred_eta  = pred[:, :, 0] * pred_mask
    pred_phi  = pred[:, :, 1] * pred_mask
    pred_pt   = pred[:, :, 2] * pred_mask

    tgt_eta   = target[:, :, 0] * target_mask
    tgt_phi   = target[:, :, 1] * target_mask
    tgt_pt    = target[:, :, 2] * target_mask

    pred_eta_s,  _ = torch.sort(pred_eta,  dim=1)
    tgt_eta_s,   _ = torch.sort(tgt_eta,   dim=1)
    pred_phi_s,  _ = torch.sort(pred_phi,  dim=1)
    tgt_phi_s,   _ = torch.sort(tgt_phi,   dim=1)
    pred_pt_s,   _ = torch.sort(pred_pt,   dim=1)
    tgt_pt_s,    _ = torch.sort(tgt_pt,    dim=1)

    l1 = nn.L1Loss()
    return (l1(pred_eta_s, tgt_eta_s) +
            l1(pred_phi_s, tgt_phi_s) +
            l1(pred_pt_s,  tgt_pt_s))


def physics_regularization(pred):
    """
    Two soft physics constraints:
    1. sum(ptrel) ≈ 1.0  — JetNet standard normalisation
    2. max(ptrel) ≤ 0.8  — soft cap on leading-particle dominance
    """
    mask      = (pred[:, :, 3] > 0.5).float()
    pt        = pred[:, :, 2] * mask

    pt_sum    = pt.sum(dim=1)
    loss_sum  = F.mse_loss(pt_sum, torch.ones_like(pt_sum))

    pt_max    = pt.max(dim=1).values
    loss_lead = F.relu(pt_max - 0.8).mean()

    return loss_sum + 0.5 * loss_lead


# ══════════════════════════════════════════════════════════════════════════════
# DATASET & COLLATE
# ══════════════════════════════════════════════════════════════════════════════

class JetGraphDataset(Dataset):
    def __init__(self, graphs, targets):
        self.graphs  = graphs
        self.targets = targets
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, idx):
        return self.graphs[idx], self.targets[idx]


def collate_fn(batch):
    return (Batch.from_data_list([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]))


# ══════════════════════════════════════════════════════════════════════════════
# AUTOENCODER TRAINING
# Joint: GNN (2560) → Bottleneck.encode (512) → ClusterDecoder → particles
#
# Loss (identical to original working AE):
#   total = wass×100 + mask_bce×2 + count_mse×0.01 + physics×0.1
# ══════════════════════════════════════════════════════════════════════════════

def train_autoencoder(gnn, bottleneck, decoder, graph_data, target_particles,
                      epochs=EPOCHS_AE, batch_size=BATCH_SIZE):
    """
    Trains GNN + DeterministicBottleneck + ClusterDecoder jointly end-to-end.

    The decoder sees 512-dim compressed latents from epoch 1, ensuring full
    compatibility at inference with no distribution mismatch.

    Returns (gnn, bottleneck, decoder) all in eval() mode.
    """
    enc_path = os.path.join(WEIGHTS_DIR, "gnn_encoder.pth")
    bn_path  = os.path.join(WEIGHTS_DIR, "bottleneck.pth")
    dec_path = os.path.join(WEIGHTS_DIR, "cluster_decoder.pth")

    # ── Already trained? ──────────────────────────────────────────────────────
    if (os.path.exists(enc_path) and os.path.exists(bn_path)
            and os.path.exists(dec_path)):
        log_print("Found trained AE weights — loading, skipping training.")
        gnn.load_state_dict(torch.load(enc_path, map_location=device))
        bottleneck.load_state_dict(torch.load(bn_path, map_location=device))
        decoder.load_state_dict(torch.load(dec_path, map_location=device))
        gnn.eval(); bottleneck.eval(); decoder.eval()
        return gnn, bottleneck, decoder

    # ── Resume? ───────────────────────────────────────────────────────────────
    start_epoch  = 0
    loss_history = []
    ckpt = load_latest_checkpoint("ae")
    if ckpt is not None:
        gnn.load_state_dict(ckpt['enc'])
        bottleneck.load_state_dict(ckpt['bn'])
        decoder.load_state_dict(ckpt['dec'])
        start_epoch  = ckpt['epoch']
        loss_history = ckpt.get('loss_history', [])
    else:
        log_print("No AE checkpoint found — training from scratch.")

    # ── DataLoader ────────────────────────────────────────────────────────────
    dataset = JetGraphDataset(graph_data, target_particles)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         collate_fn=collate_fn, num_workers=0)

    all_params = (list(gnn.parameters()) +
                  list(bottleneck.encoder.parameters()) +
                  list(decoder.parameters()))
    optimizer  = optim.AdamW(all_params, lr=1e-3, weight_decay=1e-5)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - start_epoch, eta_min=1e-6
    )

    n_batches = len(loader)
    log_print(f"\n=== AE Training: GNN → Bottleneck → ClusterDecoder ===")
    log_print(f"    epochs {start_epoch}→{epochs} | "
              f"{n_batches} batches/epoch | batch={batch_size}")
    log_print("    Loss = wass×100 + mask_bce×2 + count_mse×0.01 + physics×0.1")

    gnn.train(); bottleneck.train(); decoder.train()

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0

        for bidx, (bg, bt) in enumerate(loader):
            bg = bg.to(device)
            bt = bt.to(device)

            # Forward pass: graph → 2560 → 512 → 150 particles
            emb  = gnn(bg.x, bg.edge_index, bg.batch)   # [B, 2560]
            z    = bottleneck.encode(emb)                 # [B, 512]
            pred = decoder(z)                             # [B, 150, 4]

            # Wasserstein on η, φ, pT
            loss_w = masked_wasserstein_loss(pred, bt)

            # Mask BCE — which slots are occupied
            loss_mask = F.binary_cross_entropy(
                pred[:, :, 3].clamp(1e-6, 1 - 1e-6),
                (bt[:, :, 3] > 0.5).float()
            )

            # Count MSE — number of valid particles per jet
            loss_count = F.mse_loss(
                pred[:, :, 3].sum(dim=1),
                bt[:, :, 3].sum(dim=1)
            )

            # Physics regularization
            loss_phys = physics_regularization(pred)

            total = (loss_w     * 100.0 +
                     loss_mask  *   2.0 +
                     loss_count *   0.01 +
                     loss_phys  *   0.1)

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            epoch_loss += total.item()

            if (bidx + 1) % 100 == 0:
                log_print(f"  E{epoch+1}/{epochs} B{bidx+1}/{n_batches} | "
                          f"total={total.item():.4f}  "
                          f"wass={loss_w.item():.4f}  "
                          f"mask={loss_mask.item():.4f}  "
                          f"count={loss_count.item():.4f}  "
                          f"phys={loss_phys.item():.4f}")

        scheduler.step()
        avg = epoch_loss / n_batches
        loss_history.append(avg)
        log_print(f"AE Epoch {epoch+1:4d}/{epochs} | "
                  f"avg_loss={avg:.4f} | LR={scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint({
                'epoch':        epoch + 1,
                'enc':          gnn.state_dict(),
                'bn':           bottleneck.state_dict(),
                'dec':          decoder.state_dict(),
                'opt':          optimizer.state_dict(),
                'loss_history': loss_history,
            }, prefix="ae", epoch=epoch + 1)

    # ── Save final weights ────────────────────────────────────────────────────
    torch.save(gnn.state_dict(),        enc_path)
    torch.save(bottleneck.state_dict(), bn_path)
    torch.save(decoder.state_dict(),    dec_path)
    log_print(f"AE complete. Saved:\n  {enc_path}\n  {bn_path}\n  {dec_path}")
    cleanup_checkpoints("ae")

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('AE Joint Training (GNN → BN → ClusterDecoder)')
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, "ae_loss.png"))
    plt.close()
    log_print("AE loss plot saved.")

    gnn.eval(); bottleneck.eval(); decoder.eval()
    return gnn, bottleneck, decoder


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 4: CONDITIONAL FLOW MATCHING  (FiLM + jet-type embedding)
# Identical to V3 — unchanged.
# ══════════════════════════════════════════════════════════════════════════════

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        if t.dim() > 1:
            t = t.squeeze(-1)
        half = self.dim // 2
        freq = torch.exp(torch.arange(half, device=t.device)
                         * -(math.log(10000) / (half - 1)))
        emb  = t[:, None] * freq[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class FiLMBlock(nn.Module):
    """
    Residual block with FiLM conditioning at every layer.
    FiLM: h = (1 + scale(cond)) * LayerNorm(h) + shift(cond)
    Stronger than input-only conditioning — modulates every representation.
    """
    def __init__(self, hidden, cond_dim, dropout=0.1):
        super().__init__()
        self.norm1      = nn.LayerNorm(hidden)
        self.lin1       = nn.Linear(hidden, hidden)
        self.act        = nn.LeakyReLU(0.2)
        self.drop       = nn.Dropout(dropout)
        self.norm2      = nn.LayerNorm(hidden)
        self.lin2       = nn.Linear(hidden, hidden)
        self.t_proj     = nn.Linear(hidden, hidden)
        self.film_scale = nn.Linear(cond_dim, hidden)
        self.film_shift = nn.Linear(cond_dim, hidden)

    def forward(self, x, t_emb, cond):
        scale = 1.0 + self.film_scale(cond)
        shift = self.film_shift(cond)
        h     = self.norm1(x) * scale + shift + self.t_proj(t_emb)
        h     = self.drop(self.lin1(self.act(h)))
        h     = self.lin2(self.act(self.norm2(h)))
        return x + h


class ConditionalFlowModel(nn.Module):
    """
    Conditional flow matching:
    - Jet type: discrete embedding g/q/t → 8-dim learned vector
    - Continuous conditions: [jet_pt, jet_mass, multiplicity] normalised
    - FiLM conditioning injected at every residual block (not just input)
    """
    def __init__(self, emb_dim=BOTTLENECK_DIM, cond_dim=COND_DIM_TOTAL,
                 hidden=DIFFUSION_HIDDEN, time_dim=512, n_layers=8):
        super().__init__()
        self.type_emb   = nn.Embedding(NUM_JET_TYPES, JET_TYPE_EMBED_DIM)
        self.time_mlp   = nn.Sequential(
            SinusoidalEmbeddings(time_dim),
            nn.Linear(time_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden,   hidden),
        )
        self.input_proj = nn.Linear(emb_dim, hidden)
        self.blocks     = nn.ModuleList([
            FiLMBlock(hidden, cond_dim) for _ in range(n_layers)
        ])
        self.out_proj   = nn.Linear(hidden, emb_dim)

    def forward(self, x, t, cont_cond, type_ids):
        cond  = torch.cat([cont_cond, self.type_emb(type_ids)], dim=-1)
        t_emb = self.time_mlp(t)
        h     = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h, t_emb, cond)
        return self.out_proj(h)


def train_flow_matching(embeddings, jet_features, jet_type_ids,
                        epochs=EPOCHS_DIFF, batch_size=BATCH_SIZE_FLOW):
    """
    Train ConditionalFlowModel on normalised 512-dim bottleneck latents.
    Linear flow path: x_t = (1-t)·noise + t·data, velocity target = data - noise.
    """
    flow_path = os.path.join(WEIGHTS_DIR, "cond_flow.pth")

    emb_dim = embeddings.shape[1]
    model   = ConditionalFlowModel(emb_dim=emb_dim).to(device)
    opt     = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    sch     = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    start_epoch  = 0
    loss_history = []
    ckpt = load_latest_checkpoint("flow")
    if ckpt is not None:
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        start_epoch  = ckpt['epoch']
        loss_history = ckpt.get('loss_history', [])

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(embeddings,   dtype=torch.float32),
        torch.tensor(jet_features, dtype=torch.float32),
        torch.tensor(jet_type_ids, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    n_batches = len(loader)
    log_print(f"\n=== Flow Matching | epochs {start_epoch}→{epochs} | "
              f"{n_batches} batches/epoch | batch={batch_size} ===")

    model.train()
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        for emb_b, cond_b, type_b in loader:
            emb_b  = emb_b.to(device)
            cond_b = cond_b.to(device)
            type_b = type_b.to(device)

            t      = torch.rand(emb_b.size(0), 1, device=device)
            noise  = torch.randn_like(emb_b)
            x_t    = (1 - t) * noise + t * emb_b
            pred_v = model(x_t, t, cond_b, type_b)
            loss   = F.mse_loss(pred_v, emb_b - noise)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()

        sch.step()
        avg = epoch_loss / n_batches
        loss_history.append(avg)

        if (epoch + 1) % 10 == 0:
            log_print(f"Flow Epoch {epoch+1:4d}/{epochs} | "
                      f"loss={avg:.6f} | LR={sch.get_last_lr()[0]:.2e}")

        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint({
                'epoch':        epoch + 1,
                'model':        model.state_dict(),
                'opt':          opt.state_dict(),
                'loss_history': loss_history,
            }, prefix="flow", epoch=epoch + 1)

    torch.save(model.state_dict(), flow_path)
    log_print(f"Flow model saved: {flow_path}")
    cleanup_checkpoints("flow")

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.title('Flow Matching Loss (FiLM + JetType Embedding)')
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, "flow_loss.png"))
    plt.close()

    model.eval()
    return model


def sample_flow(model, cont_conds, type_ids, n_steps=500,
                batch_size=BATCH_SIZE_FLOW):
    """Euler integration of the learned velocity field."""
    model.eval()
    samples = []
    with torch.no_grad():
        for i in range(0, len(cont_conds), batch_size):
            bs   = min(batch_size, len(cont_conds) - i)
            cond = torch.tensor(cont_conds[i:i+bs], dtype=torch.float32).to(device)
            tids = torch.tensor(type_ids[i:i+bs],   dtype=torch.long).to(device)
            x    = torch.randn(bs, BOTTLENECK_DIM, device=device)
            dt   = 1.0 / n_steps
            for s in range(n_steps):
                t = torch.full((bs, 1), s / n_steps, device=device)
                x = x + dt * model(x, t, cond, tids)
            samples.append(x.cpu().numpy())
    return np.concatenate(samples, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION & METRICS
# ══════════════════════════════════════════════════════════════════════════════

def visualize_jets(particle_list, titles, save_name="generated"):
    n, nc = len(particle_list), 2
    nr    = (n + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(14 * nc, 6 * nr))
    if nr == 1:
        axes = np.array(axes).reshape(1, -1)
    for i, (p, t) in enumerate(zip(particle_list, titles)):
        ax   = axes[i // nc, i % nc]
        mask = p[:, 3] > 0.5
        if mask.sum():
            sc = ax.scatter(p[mask, 0], p[mask, 1], c=p[mask, 2],
                            s=50, cmap='viridis', alpha=0.7)
            plt.colorbar(sc, ax=ax, label='pT')
        ax.set(title=f"{t} (n={mask.sum()})", xlabel='η', ylabel='φ',
               xlim=(-0.8, 0.8), ylim=(-0.8, 0.8))
        ax.grid(alpha=0.3)
    for i in range(n, nr * nc):
        axes[i // nc, i % nc].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{save_name}.png"))
    plt.close()
    log_print(f"Saved: {save_name}.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main_pipeline():
    log_print("\n" + "=" * 70)
    log_print("PIPELINE V4: DistancePool + DetBottleneck + ClusterDecoder + FiLM")
    log_print("=" * 70 + "\n")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    particle_data, jet_data, jet_types_all = load_jetnet_data()
    graph_data, target_particles, original_jets, collected_types = \
        collect_graph_and_targets(particle_data, jet_data, jet_types_all)
    N = len(graph_data)

    # ── 2. AE: joint GNN + Bottleneck + ClusterDecoder ───────────────────────
    # Forward: GNN(2560) → BN.encode(512) → ClusterDecoder → particles
    # All three trained jointly — decoder adapts to 512-dim inputs from day one.
    gnn        = DistancePoolChebNet().to(device)
    bottleneck = DeterministicBottleneck().to(device)
    decoder    = ClusterDecoder(latent_dim=BOTTLENECK_DIM).to(device)

    gnn, bottleneck, decoder = train_autoencoder(
        gnn, bottleneck, decoder, graph_data, target_particles,
        epochs=EPOCHS_AE, batch_size=BATCH_SIZE
    )

    # ── 3. Extract 512-dim latents (post-training, frozen) ────────────────────
    log_print("Extracting bottleneck latents for flow training...")
    lat_cache = os.path.join(WEIGHTS_DIR, "bottleneck_latents.pt")

    # Invalidate stale cache: if AE weights are newer than the cache, the cache
    # was built from a previous model and must be regenerated.
    cache_is_stale = False
    if os.path.exists(lat_cache):
        cache_mtime  = os.path.getmtime(lat_cache)
        enc_mtime    = os.path.getmtime(enc_path) if os.path.exists(enc_path) else 0
        if enc_mtime > cache_mtime:
            log_print("  AE weights newer than latent cache — regenerating cache.")
            os.remove(lat_cache)
            cache_is_stale = True

    if os.path.exists(lat_cache):
        log_print("  Loading cached latents...")
        z_all = torch.load(lat_cache, map_location='cpu').numpy()
    else:
        gnn.eval(); bottleneck.eval()
        parts = []
        with torch.no_grad():
            for i in range(0, N, BATCH_SIZE_FLOW):
                b   = Batch.from_data_list(graph_data[i:i + BATCH_SIZE_FLOW]).to(device)
                emb = gnn(b.x, b.edge_index, b.batch)
                z   = bottleneck.encode(emb)
                parts.append(z.cpu())
        z_all = torch.cat(parts, 0).numpy()
        torch.save(torch.from_numpy(z_all), lat_cache)
    log_print(f"Latents shape: {z_all.shape}")   # [N, 512]

    # Z-score normalise — clean since encoder is deterministic
    z_mean = z_all.mean(0);  z_std = z_all.std(0) + 1e-6
    z_norm = (z_all - z_mean) / z_std
    np.save(os.path.join(WEIGHTS_DIR, "z_mean.npy"), z_mean)
    np.save(os.path.join(WEIGHTS_DIR, "z_std.npy"),  z_std)
    log_print("Latents normalised (mean≈0, std≈1)")

    # ── 4. Conditioning arrays ────────────────────────────────────────────────
    jet_pt   = jet_data[:N, 0]
    jet_mass = jet_data[:N, 2]
    jet_mult = jet_data[:N, 3] / TOTAL_PARTICLES

    def minmax(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    jet_features = np.stack([minmax(jet_pt), minmax(jet_mass), jet_mult], axis=1)
    jet_type_ids = collected_types[:N]

    log_print(f"Jet types: g={(jet_type_ids == 0).sum()}  "
              f"q={(jet_type_ids == 1).sum()}  t={(jet_type_ids == 2).sum()}")

    # ── 5. Flow matching ──────────────────────────────────────────────────────
    flow_model = train_flow_matching(
        z_norm, jet_features, jet_type_ids, epochs=EPOCHS_DIFF
    )

    # ── 6. Sample & decode ────────────────────────────────────────────────────
    log_print("Sampling generated jets...")
    idx          = np.random.choice(N, 8, replace=False)
    cond_sample  = jet_features[idx]
    types_sample = jet_type_ids[idx]

    z_gen = sample_flow(flow_model, cond_sample, types_sample, n_steps=750)
    z_gen = z_gen * z_std + z_mean   # denormalise → raw 512-dim space

    with torch.no_grad():
        # Decoder takes 512-dim directly — trained on BN outputs, matches exactly.
        z_t       = torch.tensor(z_gen, dtype=torch.float32).to(device)
        gen_parts = decoder(z_t).cpu().numpy()   # [8, 150, 4]

    type_names = {0: 'g-jet', 1: 'q-jet', 2: 't-jet'}
    titles = [f"{type_names[t]} #{i+1}" for i, t in enumerate(types_sample)]
    visualize_jets(gen_parts, titles, save_name="generated_jets_v4")

    # ── 7. Reconstruction metrics ─────────────────────────────────────────────
    log_print("\n--- Reconstruction metrics (10 jets) ---")
    n_eval = min(10, N)
    with torch.no_grad():
        eval_graphs = Batch.from_data_list(graph_data[:n_eval]).to(device)
        eval_emb    = gnn(eval_graphs.x, eval_graphs.edge_index, eval_graphs.batch)
        eval_z      = bottleneck.encode(eval_emb)
        recon       = decoder(eval_z).cpu().numpy()

    ws_eta, ws_phi, ws_pt = [], [], []
    for i in range(n_eval):
        orig = particle_data[i][particle_data[i][:, 3] == 1]
        rec  = recon[i][recon[i][:, 3] > 0.5]
        if len(orig) and len(rec):
            ws_eta.append(wasserstein_distance(orig[:, 0], rec[:, 0]))
            ws_phi.append(wasserstein_distance(orig[:, 1], rec[:, 1]))
            ws_pt.append( wasserstein_distance(orig[:, 2], rec[:, 2]))

    if ws_eta:
        log_print(f"  WS η={np.mean(ws_eta):.4f}  "
                  f"φ={np.mean(ws_phi):.4f}  "
                  f"pT={np.mean(ws_pt):.4f}")

    log_print("\nPipeline V4 complete.")
    return "Done"


if __name__ == "__main__":
    main_pipeline()
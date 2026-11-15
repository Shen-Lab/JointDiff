"""AlphaFold-Multimer training loss (PyTorch re-implementation, single example, no batch dim).

Total loss (Evans et al. 2021, "Protein complex prediction with AlphaFold-Multimer"):

    L = 0.5 * L_fape        (chain-aware backbone + side-chain FAPE, averaged over structure-module
                             iterations; see `fape_multimer`)
      + 0.3 * L_distogram   (cross-entropy on binned CB-CB distances)
      + 2.0 * L_msa         (BERT-style masked-MSA cross-entropy)
      + 0.01 * L_conf       (pLDDT + PAE + experimentally-resolved heads, all resolution-gated
                             except the resolved head)
      + 0.03 * L_viol       (fine-tuning only; NOT implemented here, pass `violation_loss`)

Differences w.r.t. monomer AlphaFold2 that matter here:
  * FAPE is split into an INTRA-chain term (clamp 10 A, length scale 10 A) and an INTER-chain /
    interface term (clamp 30 A, length scale 20 A), each normalised by its own pair count.
  * A PAE (predicted aligned error) head loss is part of the confidence loss.

Not included (they live in the data pipeline / model, not in the loss):
  * Ground-truth chain permutation / symmetric-chain alignment. The ground truth passed in is
    assumed to be ALREADY matched to the predicted chain order (the paper does this by greedy
    anchor-based chain assignment before computing the loss).
  * Side-chain atom14 frame construction. `fape_multimer` accepts arbitrary frames/points, so the
    caller builds the backbone and side-chain sets and calls it once per set.
"""

import torch
import torch.nn.functional as F

EPS = 1e-8


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def masked_mean(mask, value, eps=EPS):
    """Mean of `value` over entries where `mask` == 1."""
    return (mask * value).sum() / (mask.sum() + eps)


def softmax_cross_entropy(logits, one_hot_labels):
    """-sum_c y_c log softmax(logits)_c over the last dim."""
    return -(one_hot_labels * F.log_softmax(logits, dim=-1)).sum(-1)


def invert_apply(rot, trans, points):
    """Express global `points` [P,3] in each local frame (rot [F,3,3], trans [F,3]) -> [F,P,3].

    local = R^T (x - t)
    """
    diff = points[None, :, :] - trans[:, None, :]            # [F,P,3]
    return torch.einsum("fji,fpj->fpi", rot, diff)           # R^T @ diff


# --------------------------------------------------------------------------------------
# 1. FAPE (chain-aware)
# --------------------------------------------------------------------------------------
def _fape_pairs(pred_rot, pred_trans, pred_pts,
                true_rot, true_trans, true_pts,
                pair_mask, clamp, length_scale):
    """FAPE restricted to the (frame, point) pairs selected by `pair_mask` [F,P].

    error_ij = || T_pred_i^-1 x_pred_j  -  T_true_i^-1 x_true_j ||   (computed in local frames)
    loss     = mean_{ij in mask} min(error_ij, clamp) / length_scale
    """
    local_pred = invert_apply(pred_rot, pred_trans, pred_pts)
    local_true = invert_apply(true_rot, true_trans, true_pts)
    err = torch.sqrt(((local_pred - local_true) ** 2).sum(-1) + 1e-4)  # [F,P]
    if clamp is not None:
        err = err.clamp(max=clamp)
    err = err / length_scale
    return masked_mean(pair_mask, err)


def fape_multimer(pred_rot, pred_trans, pred_pts,
                  true_rot, true_trans, true_pts,
                  frame_mask, point_mask,
                  frame_chain, point_chain,
                  intra_clamp=10.0, intra_scale=10.0,
                  inter_clamp=30.0, inter_scale=20.0,
                  use_clamp=True):
    """AF-Multimer FAPE = intra-chain FAPE + interface FAPE.

    Args:
      pred_rot/true_rot     [F,3,3]  frame rotations
      pred_trans/true_trans [F,3]    frame origins
      pred_pts/true_pts     [P,3]    points evaluated in the frames (CA for backbone FAPE,
                                     all atoms for side-chain FAPE)
      frame_mask [F], point_mask [P] 1 where ground truth exists
      frame_chain [F], point_chain [P] integer chain (asym) ids
      use_clamp: AF2 disables clamping for ~10% of samples; sample this outside and pass here.
    """
    same_chain = (frame_chain[:, None] == point_chain[None, :]).float()    # [F,P]
    valid = frame_mask[:, None] * point_mask[None, :]
    args = (pred_rot, pred_trans, pred_pts, true_rot, true_trans, true_pts)

    # Within a chain: tight clamp, so large intra-chain errors saturate quickly.
    intra = _fape_pairs(*args, valid * same_chain,
                        intra_clamp if use_clamp else None, intra_scale)
    # Between chains: looser clamp / larger scale, because relative docking errors are
    # naturally bigger and we still want gradient signal for them.
    inter = _fape_pairs(*args, valid * (1.0 - same_chain),
                        inter_clamp if use_clamp else None, inter_scale)
    return intra + inter


def structure_loss(pred_frames_per_iter, true_frames, point_sets, use_clamp=True):
    """Average chain-aware FAPE over all structure-module iterations (auxiliary losses).

    Args:
      pred_frames_per_iter: list over iterations of dicts {"bb": (rot,trans,pts), "sc": (rot,trans,pts)}
      true_frames: dict {"bb": (rot,trans,pts), "sc": (rot,trans,pts)}
      point_sets : dict {"bb": (frame_mask,point_mask,frame_chain,point_chain), "sc": (...)}
    Returns scalar: 0.5*bb + 0.5*sc style mixing is left to the caller via weights below.
    """
    losses = []
    for pred in pred_frames_per_iter:
        per_set = []
        for key in ("bb", "sc"):
            fm, pm, fc, pc = point_sets[key]
            per_set.append(fape_multimer(*pred[key], *true_frames[key],
                                         fm, pm, fc, pc, use_clamp=use_clamp))
        losses.append(0.5 * per_set[0] + 0.5 * per_set[1])  # backbone / side-chain mix
    return torch.stack(losses).mean()


# --------------------------------------------------------------------------------------
# 2. Distogram loss
# --------------------------------------------------------------------------------------
def distogram_loss(logits, true_cb, cb_mask, min_bin=2.3125, max_bin=21.6875, num_bins=64):
    """Cross-entropy between predicted distance histogram and the true CB-CB distance bin.

    logits [N,N,num_bins]; true_cb [N,3] (CA for glycine); cb_mask [N].
    Distances beyond the last bin edge land in the final bin.
    """
    d2 = ((true_cb[:, None] - true_cb[None, :]) ** 2).sum(-1, keepdim=True)  # [N,N,1]
    edges = torch.linspace(min_bin, max_bin, num_bins - 1, device=logits.device) ** 2
    labels = (d2 > edges).sum(-1)                                            # [N,N] in [0,63]
    ce = softmax_cross_entropy(logits, F.one_hot(labels, num_bins).float())
    pair_mask = cb_mask[:, None] * cb_mask[None, :]
    # AF2 normalises per-row then averages rows; this is the equivalent global masked mean.
    return masked_mean(pair_mask, ce)


# --------------------------------------------------------------------------------------
# 3. Masked-MSA loss
# --------------------------------------------------------------------------------------
def masked_msa_loss(logits, true_msa, bert_mask, num_classes=23):
    """BERT-style loss: predict original MSA tokens at positions that were masked/corrupted.

    logits [S,N,num_classes]; true_msa [S,N] int; bert_mask [S,N] (1 = position was masked).
    """
    ce = softmax_cross_entropy(logits, F.one_hot(true_msa, num_classes).float())
    return masked_mean(bert_mask, ce)


# --------------------------------------------------------------------------------------
# 4. Confidence heads
# --------------------------------------------------------------------------------------
def plddt_loss(logits, pred_ca, true_ca, ca_mask, resolution,
               cutoff=15.0, num_bins=50, res_range=(0.1, 3.0)):
    """Train pLDDT head to predict the true per-residue CA-lDDT (binned, 50 bins in [0,1]).

    logits [N,num_bins]; pred_ca/true_ca [N,3]; ca_mask [N]. Target is stop-gradient.
    Only used when the experimental resolution is in `res_range`.
    """
    pred_ca, true_ca = pred_ca.detach(), true_ca.detach()
    d_pred = torch.cdist(pred_ca, pred_ca)
    d_true = torch.cdist(true_ca, true_ca)
    n = true_ca.shape[0]
    # Neighbours: true distance < cutoff, excluding self, both residues resolved.
    nbr = ((d_true < cutoff).float() * ca_mask[:, None] * ca_mask[None, :]
           * (1.0 - torch.eye(n, device=true_ca.device)))
    diff = (d_pred - d_true).abs()
    score = 0.25 * sum((diff < t).float() for t in (0.5, 1.0, 2.0, 4.0))
    lddt = (nbr * score).sum(-1) / (nbr.sum(-1) + EPS)                     # [N]
    labels = (lddt * num_bins).floor().long().clamp(max=num_bins - 1)
    ce = softmax_cross_entropy(logits, F.one_hot(labels, num_bins).float())
    loss = masked_mean(ca_mask, ce)
    ok = ((resolution >= res_range[0]) & (resolution <= res_range[1])).float()
    return loss * ok


def pae_loss(logits, pred_rot, pred_trans, pred_ca, true_rot, true_trans, true_ca,
             frame_mask, resolution, num_bins=64, max_error=31.0, res_range=(0.1, 3.0)):
    """Predicted-aligned-error head loss (added in AF-Multimer for ipTM/ranking).

    Aligned error of residue j when aligned on residue i's frame:
        e_ij = || T_pred_i^-1 x_pred_j - T_true_i^-1 x_true_j ||
    logits [N,N,num_bins]; bins are uniform in [0, max_error], last bin catches overflow.
    """
    local_pred = invert_apply(pred_rot.detach(), pred_trans.detach(), pred_ca.detach())
    local_true = invert_apply(true_rot, true_trans, true_ca)
    err = torch.sqrt(((local_pred - local_true) ** 2).sum(-1) + 1e-8)       # [N,N]
    edges = torch.linspace(0.0, max_error, num_bins - 1, device=logits.device)
    labels = (err[..., None] > edges).sum(-1)
    ce = softmax_cross_entropy(logits, F.one_hot(labels, num_bins).float())
    pair_mask = frame_mask[:, None] * frame_mask[None, :]
    ok = ((resolution >= res_range[0]) & (resolution <= res_range[1])).float()
    return masked_mean(pair_mask, ce) * ok


def experimentally_resolved_loss(logits, atom37_resolved, atom37_exists, resolution,
                                 res_range=(0.1, 3.0)):
    """Binary CE: was each atom actually observed in the experiment? logits/targets [N,37]."""
    bce = F.binary_cross_entropy_with_logits(logits, atom37_resolved.float(), reduction="none")
    loss = masked_mean(atom37_exists, bce)
    ok = ((resolution >= res_range[0]) & (resolution <= res_range[1])).float()
    return loss * ok


# --------------------------------------------------------------------------------------
# 5. Total loss
# --------------------------------------------------------------------------------------
def alphafold_multimer_loss(out, batch, w_fape=0.5, w_dist=0.3, w_msa=2.0, w_conf=0.01,
                            w_viol=0.03, violation_loss=None, use_clamp=True):
    """Combine everything. `out` = model outputs, `batch` = ground truth / masks.

    Expected keys (all single-example, no batch dim):
      out:   frames_per_iter (list for structure_loss), distogram_logits [N,N,64],
             masked_msa_logits [S,N,23], plddt_logits [N,50], pae_logits [N,N,64],
             resolved_logits [N,37], final_rot [N,3,3], final_trans [N,3], final_ca [N,3]
      batch: true_frames, point_sets (see `structure_loss`), true_cb, cb_mask, true_msa,
             bert_mask, true_ca, ca_mask, true_rot, true_trans, atom37_resolved,
             atom37_exists, resolution (0-dim tensor)
    """
    l_fape = structure_loss(out["frames_per_iter"], batch["true_frames"],
                            batch["point_sets"], use_clamp)
    l_dist = distogram_loss(out["distogram_logits"], batch["true_cb"], batch["cb_mask"])
    l_msa = masked_msa_loss(out["masked_msa_logits"], batch["true_msa"], batch["bert_mask"])
    l_conf = (plddt_loss(out["plddt_logits"], out["final_ca"], batch["true_ca"],
                         batch["ca_mask"], batch["resolution"])
              + pae_loss(out["pae_logits"], out["final_rot"], out["final_trans"],
                         out["final_ca"], batch["true_rot"], batch["true_trans"],
                         batch["true_ca"], batch["ca_mask"], batch["resolution"])
              + experimentally_resolved_loss(out["resolved_logits"], batch["atom37_resolved"],
                                             batch["atom37_exists"], batch["resolution"]))
    total = w_fape * l_fape + w_dist * l_dist + w_msa * l_msa + w_conf * l_conf
    if violation_loss is not None:           # fine-tuning stage only
        total = total + w_viol * violation_loss
    return total, {"fape": l_fape, "distogram": l_dist, "msa": l_msa, "confidence": l_conf}


# --------------------------------------------------------------------------------------
# Smoke test with random data
# --------------------------------------------------------------------------------------
def _rand_rot(n):
    q, _ = torch.linalg.qr(torch.randn(n, 3, 3))
    return q * torch.sign(torch.linalg.det(q))[:, None, None]


if __name__ == "__main__":
    torch.manual_seed(0)
    N, S = 20, 8
    chain = torch.tensor([0] * 12 + [1] * 8)
    ones = torch.ones(N)

    def frames():
        return (_rand_rot(N), torch.randn(N, 3) * 10, torch.randn(N, 3) * 10)

    true_bb = frames()
    ps = (ones, ones, chain, chain)
    out = {
        "frames_per_iter": [{"bb": frames(), "sc": frames()} for _ in range(3)],
        "distogram_logits": torch.randn(N, N, 64),
        "masked_msa_logits": torch.randn(S, N, 23),
        "plddt_logits": torch.randn(N, 50),
        "pae_logits": torch.randn(N, N, 64),
        "resolved_logits": torch.randn(N, 37),
        "final_rot": _rand_rot(N), "final_trans": torch.randn(N, 3) * 10,
        "final_ca": torch.randn(N, 3) * 10,
    }
    batch = {
        "true_frames": {"bb": true_bb, "sc": frames()},
        "point_sets": {"bb": ps, "sc": ps},
        "true_cb": torch.randn(N, 3) * 10, "cb_mask": ones,
        "true_msa": torch.randint(0, 23, (S, N)), "bert_mask": (torch.rand(S, N) < 0.15).float(),
        "true_ca": true_bb[2], "ca_mask": ones,
        "true_rot": true_bb[0], "true_trans": true_bb[1],
        "atom37_resolved": torch.randint(0, 2, (N, 37)), "atom37_exists": torch.ones(N, 37),
        "resolution": torch.tensor(2.0),
    }
    total, parts = alphafold_multimer_loss(out, batch)
    print("total", float(total), {k: round(float(v), 3) for k, v in parts.items()})

    # Sanity: perfect prediction -> FAPE ~ 0 (only the 1e-4 eps floor remains).
    perfect = fape_multimer(*true_bb, *true_bb, ones, ones, chain, chain)
    print("FAPE(perfect) =", float(perfect))


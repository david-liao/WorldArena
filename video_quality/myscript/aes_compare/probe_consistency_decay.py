"""Probe whether background_consistency degradation in `improved` is consistent
with a "first-frame distribution shift" hypothesis.

Three measurements (per episode, paired origin <-> improved):

  M1. ||F0_improved - F0_origin||   — how far the new first frame sits from
      the training-style first frame in CLIP feature space.
  M2. mean cos(F_t, F_0) over t≥1   — the `sim_fir` term that the metric uses.
      Lower in `improved` means subsequent frames drift away from F0 faster.
  M3. mean cos(F_t, F_{t-1})        — the `sim_pre` term. Should be ~unchanged
      because it's a local (1/24s) delta and not affected by F0 itself.
"""
import os
import sys
import torch
import torch.nn.functional as F
import clip
from PIL import Image

sys.path.insert(0, "/mydir/code/WorldArena/video_quality")
from WorldArena.utils import clip_transform, load_video

CLIP_PATH = "/mydir/code/WorldArena/video_quality/models/clip_model/ViT-B-32.pt"
ROOT = "/mydir/code/WorldArena/datasets/aesthetic_quality_comparison_3/_aes_eval_full"
SUBSETS = ["origin", "improved"]
EPS = [f"episode{k}" for k in range(1, 11)]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"loading CLIP ViT-B/32 ...")
model, _ = clip.load(CLIP_PATH, device=device)
model.eval()
tx = clip_transform(224)


def encode_video(video_dir):
    images = load_video(video_dir)            # [T, 3, H, W] uint8 tensor
    images = tx(images).to(device)
    with torch.no_grad():
        feats = model.encode_image(images).float()
        feats = F.normalize(feats, dim=-1, p=2)
    return feats   # [T, 768]


# Cache features
feats_by_subset_ep = {}
for sub in SUBSETS:
    for ep in EPS:
        vd = os.path.join(ROOT, sub, "generated_dataset", "task_0", f"ep_{ep}", "gid_0", "video")
        feats_by_subset_ep[(sub, ep)] = encode_video(vd)
        print(f"  encoded {sub}/{ep}: {feats_by_subset_ep[(sub, ep)].shape}")

print()
print("=" * 92)
print("Per-episode probes (CLIP ViT-B/32 = same backbone as background_consistency)")
print("=" * 92)
print(f"{'episode':<10}"
      f" {'M1_dist(F0_imp,F0_orig)':>24}"
      f" {'sim_fir orig':>13}  {'sim_fir imp':>13}  {'Δ_fir':>9}"
      f" {'sim_pre orig':>13}  {'sim_pre imp':>13}  {'Δ_pre':>9}")
print("-" * 130)

m1, dfir, dpre = [], [], []
for ep in EPS:
    f_o = feats_by_subset_ep[("origin", ep)]
    f_i = feats_by_subset_ep[("improved", ep)]

    # M1: distance between the two first frames (in cosine-distance, 1 - cos)
    dist01 = 1.0 - F.cosine_similarity(f_o[0:1], f_i[0:1]).item()

    # M2: sim_fir = mean_t>=1 cos(F_t, F_0)
    sim_fir_o = F.cosine_similarity(f_o[1:], f_o[0:1].expand_as(f_o[1:])).clamp_min(0).mean().item()
    sim_fir_i = F.cosine_similarity(f_i[1:], f_i[0:1].expand_as(f_i[1:])).clamp_min(0).mean().item()

    # M3: sim_pre = mean_t>=1 cos(F_t, F_{t-1})
    sim_pre_o = F.cosine_similarity(f_o[1:], f_o[:-1]).clamp_min(0).mean().item()
    sim_pre_i = F.cosine_similarity(f_i[1:], f_i[:-1]).clamp_min(0).mean().item()

    d_fir = sim_fir_i - sim_fir_o
    d_pre = sim_pre_i - sim_pre_o
    m1.append(dist01)
    dfir.append(d_fir)
    dpre.append(d_pre)

    print(f"{ep:<10}"
          f" {dist01:>24.4f}"
          f" {sim_fir_o:>13.4f}  {sim_fir_i:>13.4f}  {d_fir:>+9.4f}"
          f" {sim_pre_o:>13.4f}  {sim_pre_i:>13.4f}  {d_pre:>+9.4f}")

print("-" * 130)
print(f"{'mean':<10} {sum(m1)/len(m1):>24.4f}"
      f" {'':>13}  {'':>13}  {sum(dfir)/len(dfir):>+9.4f}"
      f" {'':>13}  {'':>13}  {sum(dpre)/len(dpre):>+9.4f}")

print()
print("Interpretation:")
print(f"  M1 mean  = {sum(m1)/len(m1):.4f}  (pure CLIP cosine-distance between F0_improved and F0_origin)")
print(f"  Δ sim_fir mean = {sum(dfir)/len(dfir):+.4f}  (negative = improved drifts away from F0 faster)")
print(f"  Δ sim_pre mean = {sum(dpre)/len(dpre):+.4f}  (~0 = local frame-to-frame motion unchanged)")
print()
n_neg_fir = sum(1 for d in dfir if d < 0)
n_neg_pre = sum(1 for d in dpre if d < 0)
print(f"  sim_fir down in improved on {n_neg_fir}/10 episodes")
print(f"  sim_pre down in improved on {n_neg_pre}/10 episodes")

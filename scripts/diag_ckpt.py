"""Diagnostic: compare init / raw / EMA checkpoints to find why samples are noise."""
import torch as th

LOG = "/app/logs/anime-aligned-curated"

paths = {
    "init": f"{LOG}/model000000.pt",
    "raw30k": f"{LOG}/model030000.pt",
    "ema30k": f"{LOG}/ema_0.9999_030000.pt",
    "ema0": f"{LOG}/ema_0.9999_000000.pt",
}

sds = {k: th.load(v, map_location="cpu") for k, v in paths.items()}

keys = list(sds["init"].keys())
print(f"num tensors: {len(keys)}")

# 1) finite check + global norms
for k, sd in sds.items():
    tot, bad = 0.0, 0
    for name in keys:
        t = sd[name].float()
        tot += t.pow(2).sum().item()
        if not th.isfinite(t).all():
            bad += 1
    print(f"{k:7s} global L2 = {tot ** 0.5:12.4f}   non-finite tensors: {bad}")

# 2) how much of the random init still survives in each checkpoint
#    ema_t = a * init + (1-a) * trajectory-average, with a = 0.9999**30000 = e^-3 = 0.0498
def rel_dist(a, b):
    num, den = 0.0, 0.0
    for name in keys:
        x, y = sds[a][name].float(), sds[b][name].float()
        num += (x - y).pow(2).sum().item()
        den += y.pow(2).sum().item()
    return (num / den) ** 0.5

print()
print(f"||ema0   - init  || / ||init||   = {rel_dist('ema0', 'init'):.6f}  (should be ~0)")
print(f"||raw30k - init  || / ||init||   = {rel_dist('raw30k', 'init'):.6f}")
print(f"||ema30k - raw30k|| / ||raw30k|| = {rel_dist('ema30k', 'raw30k'):.6f}")

# 3) project (ema30k - raw30k) onto (init - raw30k): if EMA is dragged toward init,
#    the cosine will be strongly positive and the coefficient ~= 0.9999**30000.
num, d1, d2 = 0.0, 0.0, 0.0
for name in keys:
    e = (sds["ema30k"][name].float() - sds["raw30k"][name].float()).flatten()
    i = (sds["init"][name].float() - sds["raw30k"][name].float()).flatten()
    num += th.dot(e, i).item()
    d1 += e.pow(2).sum().item()
    d2 += i.pow(2).sum().item()
print(f"\ncos(ema30k-raw30k, init-raw30k)   = {num / (d1 ** 0.5 * d2 ** 0.5):.4f}")
print(f"implied init coefficient in EMA  = {num / d2:.4f}   (0.9999^30000 = 0.0498)")

# 4) per-tensor: which layers differ most between ema and raw
diffs = []
for name in keys:
    x, y = sds["ema30k"][name].float(), sds["raw30k"][name].float()
    d = (x - y).norm().item() / (y.norm().item() + 1e-12)
    diffs.append((d, name))
diffs.sort(reverse=True)
print("\ntop-10 layers by relative EMA-vs-raw divergence:")
for d, name in diffs[:10]:
    print(f"  {d:8.4f}  {name}")

# 5) how far did each layer move from init during training (raw)
moved = []
for name in keys:
    x, y = sds["raw30k"][name].float(), sds["init"][name].float()
    moved.append(((x - y).norm().item() / (y.norm().item() + 1e-12), name))
moved.sort()
print("\n10 layers that moved LEAST from init (raw30k):")
for d, name in moved[:10]:
    print(f"  {d:8.4f}  {name}")

"""Remove the random-init component that EMA still carries at step t.

update_ema does  ema <- r*ema + (1-r)*theta,  initialized at ema_0 = theta_0.
Unrolling:       ema_t = r^t * theta_0 + (1 - r^t) * A_t
where A_t is the normalized exponentially-weighted average of the trajectory.
Since theta_0 (model000000.pt) was saved, A_t can be recovered exactly:
     A_t = (ema_t - r^t * theta_0) / (1 - r^t)
"""
import torch as th

LOG = "/app/logs/anime-aligned-curated"
r, t = 0.9999, 30000
w = r ** t  # weight the random init still holds
print(f"r^t = {w:.6f}  -> {w * 100:.2f}% of ema_{t} is still the random initialization")

ema = th.load(f"{LOG}/ema_0.9999_0{t}.pt", map_location="cpu")
init = th.load(f"{LOG}/model000000.pt", map_location="cpu")

out = {}
for k in ema:
    e, i = ema[k].float(), init[k].float()
    out[k] = ((e - w * i) / (1.0 - w)).to(ema[k].dtype)

th.save(out, f"{LOG}/ema_debiased_0{t}.pt")
print(f"wrote {LOG}/ema_debiased_0{t}.pt")

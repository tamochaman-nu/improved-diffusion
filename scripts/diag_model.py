"""Diagnostic: is the 30k model actually predicting eps well, and what does it sample?"""
import argparse, os
import numpy as np
import torch as th
from PIL import Image

from improved_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from improved_diffusion.image_datasets import load_data

LOG = "/app/logs/anime-aligned-curated"

FLAGS = dict(
    image_size=256, num_channels=256, num_res_blocks=2,
    attention_resolutions="32,16,8", learn_sigma=True, class_cond=False,
    use_checkpoint=False, diffusion_steps=1000, noise_schedule="linear",
)

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="ema_0.9999_030000.pt")
p.add_argument("--nsample", type=int, default=4)
p.add_argument("--respace", default="")
args = p.parse_args()

d = model_and_diffusion_defaults()
d.update(FLAGS)
d["timestep_respacing"] = args.respace
model, diffusion = create_model_and_diffusion(**d)
sd = th.load(os.path.join(LOG, args.ckpt), map_location="cpu")
model.load_state_dict(sd)
model.to("cuda").eval()

tag = args.ckpt.replace(".pt", "")
print(f"=== {args.ckpt} ===")

# --- A) magnitude of the zero-initialized output head -------------------------
for name in ["out.2.weight", "out.2.bias"]:
    print(f"  {name:16s} L2 = {sd[name].float().norm().item():.6f}")

# --- B) eps-prediction quality on REAL data, per timestep ---------------------
data = load_data(data_dir="/app/data", batch_size=8, image_size=256, class_cond=False)
batch, _ = next(data)
x0 = batch.to("cuda")
print(f"  data range: [{x0.min().item():.3f}, {x0.max().item():.3f}]  mean {x0.mean().item():.3f}  std {x0.std().item():.3f}")

print(f"\n  {'t':>5} {'eps_mse':>10} {'trivial':>10} {'x0_rmse':>9} {'|eps_pred|':>11} {'sigma':>9}")
th.manual_seed(0)
with th.no_grad():
    for t_val in [0, 50, 100, 250, 500, 750, 900, 975, 999]:
        t = th.full((x0.shape[0],), t_val, device="cuda", dtype=th.long)
        noise = th.randn_like(x0)
        xt = diffusion.q_sample(x0, t, noise=noise)
        out = model(xt, diffusion._scale_timesteps(t))
        eps, var = th.split(out, 3, dim=1)
        mse = (eps - noise).pow(2).mean().item()
        # trivial baseline: predict eps = x_t (optimal-ish at very high t)
        triv = (xt - noise).pow(2).mean().item()
        pred_x0 = diffusion._predict_xstart_from_eps(xt, t, eps).clamp(-1, 1)
        x0_rmse = (pred_x0 - x0).pow(2).mean().sqrt().item()
        # decode learned sigma the same way the sampler does
        pv = diffusion.p_mean_variance(model, xt, t, clip_denoised=True)
        sigma = pv["variance"].sqrt().mean().item()
        print(f"  {t_val:5d} {mse:10.5f} {triv:10.5f} {x0_rmse:9.4f} {eps.abs().mean().item():11.4f} {sigma:9.5f}")

# --- C) actually sample ------------------------------------------------------
print("\n  sampling...")
th.manual_seed(1234)
with th.no_grad():
    s = diffusion.p_sample_loop(model, (args.nsample, 3, 256, 256), clip_denoised=True, progress=True)
print(f"  sample stats: min {s.min().item():.3f} max {s.max().item():.3f} mean {s.mean().item():.3f} std {s.std().item():.3f}")
# neighbouring-pixel correlation: real images ~0.95+, pure noise ~0
d1 = (s[:, :, 1:, :] - s[:, :, :-1, :]).pow(2).mean().item()
print(f"  mean sq vertical neighbour diff: {d1:.4f}  (pure noise ~ 2*var)")

img = ((s + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(0, 2, 3, 1).cpu().numpy()
grid = np.concatenate(list(img), axis=1)
Image.fromarray(grid).save(f"{LOG}/diag_{tag}{'_' + args.respace if args.respace else ''}.png")
print(f"  wrote {LOG}/diag_{tag}.png")

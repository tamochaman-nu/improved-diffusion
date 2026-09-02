"""Arrange the sample array produced by image_sample.py into a single PNG grid."""
import argparse
import math
import os

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz_path")
    p.add_argument("--out", default=None, help="defaults to <npz_path> with .png extension")
    p.add_argument("--ncols", type=int, default=0, help="0 = auto (near-square grid)")
    args = p.parse_args()

    arr = np.load(args.npz_path)["arr_0"]
    n, h, w, c = arr.shape

    ncols = args.ncols or math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    grid = np.zeros((nrows * h, ncols * w, c), dtype=arr.dtype)
    for i, img in enumerate(arr):
        r, col = divmod(i, ncols)
        grid[r * h : (r + 1) * h, col * w : (col + 1) * w] = img

    out_path = args.out or os.path.splitext(args.npz_path)[0] + ".png"
    Image.fromarray(grid).save(out_path)
    print(f"wrote {out_path} ({n} samples, {nrows}x{ncols} grid)")


if __name__ == "__main__":
    main()

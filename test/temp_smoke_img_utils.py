"""Smoke test for ImgUtils.gaussian_blur + radial_gradient additions."""
import numpy as np
from pythontk.img_utils._img_utils import ImgUtils as I

g = I.radial_gradient((100, 50), center=(0.5, 1.0), falloff_power=0.8)
print(f"shape={g.shape} dtype={g.dtype} range=[{g.min():.3f},{g.max():.3f}]")
print(f"bottom-centre (brightest)  = {g[49, 50]:.3f}")
print(f"top-centre  (dimmest)      = {g[0, 50]:.3f}")
print(f"bottom-left (offset)       = {g[49, 0]:.3f}")

mask = (np.random.rand(64, 64) * 255).astype(np.uint8)
blurred = I.gaussian_blur(mask, radius=4)
print(f"blur in dtype={mask.dtype} -> out dtype={blurred.dtype} shape={blurred.shape}")

nb = I.gaussian_blur(mask, radius=0)
print(f"radius=0 returns unchanged: equal={ (nb==mask).all() }  is_copy={nb is not mask}")

g8 = I.radial_gradient((10, 10), center=(0.5, 0.5), dtype=np.uint8)
print(f"uint8 gradient peak = {g8.max()} (should be ~255)")

# RGBA single-channel blur
rgba = np.zeros((32, 32, 4), dtype=np.uint8)
rgba[..., 0] = 200  # red
rgba[..., 3] = 128  # mid alpha
out = I.gaussian_blur(rgba, radius=3, channel="A")
print(f"channel='A' preserved R: {(out[..., 0] == 200).all()}  alpha changed: {not (out[..., 3] == 128).all()}")

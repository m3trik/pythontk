from pythontk.img_utils._img_utils import ImgUtils
from PIL import Image

# Check bit_depth mapping
bit_depth_mapping = {v: k for k, v in ImgUtils.bit_depth.items()}
print(f"8-bit maps to: {bit_depth_mapping.get(8)}")

# Check set_bit_depth behavior for L
img = Image.new("L", (64, 64), 128)
print(f"Original mode: {img.mode}")

# Mock map_type that expects L (Roughness)
# Roughness -> L
try:
    img_out = ImgUtils.set_bit_depth(img, "Roughness")
    print(f"Roughness (L) -> {img_out.mode}")
except Exception as e:
    print(f"Error: {e}")

# Mock map_type that expects RGB (Base_Color)
# Base_Color -> RGB
try:
    img_out = ImgUtils.set_bit_depth(img, "Base_Color")
    print(f"Base_Color (RGB) -> {img_out.mode}")
except Exception as e:
    print(f"Error: {e}")

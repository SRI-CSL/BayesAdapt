from bayesadapt.datasets.slake import SLAKE
import numpy as np
from PIL import Image

def add_gaussian_noise(img: Image.Image, sigma: float, per_channel=False, rng=None):
    """
    img: PIL RGB image
    sigma: std-dev in pixel units (0..255)
    per_channel: if True, independent noise per channel; else same noise across channels. Set to False for grayscale noise.
    """
    if rng is None:
        rng = np.random.default_rng()

    arr = np.asarray(img).astype(np.float32)  # (H, W, 3) in 0..255

    if sigma <= 0:
        return img

    if per_channel:
        noise = rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    else:
        noise2d = rng.normal(0.0, sigma, size=arr.shape[:2]).astype(np.float32)
        noise = noise2d[..., None]

    noisy_image = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

dataset = SLAKE()

img = dataset[0]['image']  # PIL Image
sigma_range = [0, 1, 2, 4, 8, 16, 32, 64, 128]
noisy_images = []
for sigma in sigma_range:
    noisy_img = add_gaussian_noise(img, sigma=sigma, per_channel=False)
    noisy_images.append(noisy_img)

#concatenate images horizontally for visualization
widths, heights = zip(*(i.size for i in noisy_images))
total_width = sum(widths)
max_height = max(heights)
combined_img = Image.new('RGB', (total_width, max_height))
x_offset = 0
for im in noisy_images:
    combined_img.paste(im, (x_offset,0))
    x_offset += im.size[0]
combined_img.save('noisy_images.png')



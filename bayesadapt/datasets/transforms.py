import numpy as np
from PIL import Image

class Pipeline:    
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image)
        return image

class Resize:
    def __init__(self, max_size=224):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        scaling_factor = min(self.max_size / width, self.max_size / height)
        if scaling_factor == 1.0:
            return image  # No resizing needed
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        return image.resize((new_width, new_height), Image.LANCZOS)

class AddGaussianNoise:
    def __init__(self, sigma=0, per_channel=False, rng=None):
        self.sigma = sigma
        self.per_channel = per_channel
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()

    def __call__(self, image):
        if self.sigma <= 0:
            return image
        arr = np.asarray(image).astype(np.float32)  # (H, W, 3) in 0..255
        if self.per_channel:
            noise = self.rng.normal(0.0, self.sigma, size=arr.shape).astype(np.float32)
        else:
            noise2d = self.rng.normal(0.0, self.sigma, size=arr.shape[:2]).astype(np.float32)
            noise = noise2d[..., None]
        noisy_image = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image)



# def add_gaussian_noise(img: Image.Image, sigma: float, per_channel=False, rng=None):
    # """
    # img: PIL RGB image
    # sigma: std-dev in pixel units (0..255)
    # per_channel: if True, independent noise per channel; else same noise across channels. Set to False for grayscale noise.
    # """
    # if rng is None:
        # rng = np.random.default_rng()

    # arr = np.asarray(img).astype(np.float32)  # (H, W, 3) in 0..255

    # if sigma <= 0:
        # return img

    # if per_channel:
        # noise = rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    # else:
        # noise2d = rng.normal(0.0, sigma, size=arr.shape[:2]).astype(np.float32)
        # noise = noise2d[..., None]

    # noisy_image = np.clip(arr + noise, 0, 255).astype(np.uint8)
    # return Image.fromarray(noisy_image)

# def resize_down_only(image, max_size=224):
    # width, height = image.size
    # scaling_factor = min(1.0, max_size / width, max_size / height)
    
    # If factor is 1, no resizing is needed
    # if scaling_factor == 1.0:
        # return image
    
    # new_width = int(width * scaling_factor)
    # new_height = int(height * scaling_factor)
    # return image.resize((new_width, new_height), Image.LANCZOS)

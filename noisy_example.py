from bayesadapt.datasets.slake import SLAKE
import matplotlib.pyplot as plt

noise_stds = [0,1,2,4,8,16,32,64,128]

images = []
for noise_std in noise_stds:
    ds = SLAKE(noise_std=noise_std, split="train")
    images.append(ds[0]['image'])

#put the images in a grid and save
fig, axes = plt.subplots(1, len(noise_stds), figsize=(20, 5))
for ax, img, noise_std in zip(axes, images, noise_stds):
    ax.imshow(img)
    ax.set_title(r'$\sigma$ = ' + str(noise_std))
    ax.axis('off')
plt.tight_layout()
plt.savefig("noisy_images.png",bbox_inches='tight')

import os
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchio as tio
import tifffile as tif
import numpy as np
import gc
from natsort import natsorted

class ImageMaskDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        """
        Args:
            images_dir (str): Path to the directory containing images.
            masks_dir (str): Path to the directory containing masks.
            transform (callable, optional): Transform to apply to both images and masks.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.image_paths = sorted(glob(os.path.join(images_dir, "*_image_*.tif")))
        self.mask_paths = sorted(glob(os.path.join(masks_dir, "*_mask_*.tif")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # image = np.array(tio.ScalarImage(image_path).numpy(), dtype=np.float32)
        image = np.array(tif.imread(image_path), dtype=np.float32)
        #np.array(tif.imread(image_path)[None, ...], dtype=np.float32)
        
        # orig_image = image.copy()
        # mask = np.array(tio.LabelMap(mask_path).numpy(), dtype=np.float32)
        mask = np.array(tif.imread(mask_path), dtype=np.float32)
        #np.array(tif.imread(mask_path)[None, ...], dtype=np.float32)
        
        # print("LOADING", image.shape, mask.shape)
        # orig_mask = mask.copy()
        
        # Apply transforms
        if self.transform:
            # subject = tio.Subject(
            #     image=tio.ScalarImage(tensor=image),
            #     mask=tio.LabelMap(tensor=mask),
            # )
            # transformed = self.transform(subject)
            # image = transformed.image.data
            # mask = transformed.mask.data

            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            # np.save("/results/check.npy", image)
            # np.save("/results/mask.npy", mask)

        image_max = image.max()
        image_min = image.min()
        image = (image - image_min) / (image_max - image_min)

        if np.isnan(image).any():
            np.save("/results/nan.npy", image)
            print(f"Image nan: {image_path}")
            exit()
        
        # print(image_max, image_min)
        # print(image)
        # np.save("/results/check.npy", image)
        # s
        
        
        return image[None, ...], mask[None, ...] #, orig_image, orig_mask

def get_transforms():
    return tio.Compose([
        tio.RandomFlip(axes=(0, 1, 2), p=0.5),  # Randomly flip along x, y, z
        tio.RandomAffine(scales=(0.9, 1.1), degrees=10, isotropic=True, p=0.5),
        tio.RandomElasticDeformation(num_control_points=7, max_displacement=5.0, p=0.5),
        tio.ZNormalization(),
        # tio.RescaleIntensity(out_min_max=(0, 1)),  # Normalize intensity values
    ])

def main():
    images_dir = "/scratch/dataset_patch_128_steps_128/test/images"
    masks_dir = "/scratch/dataset_patch_128_steps_128/test/masks"
    
    dataset = ImageMaskDataset(images_dir, masks_dir, transform=get_transforms())
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)

    for images, masks in dataloader:
        print("Batch of images shape:", images.shape)
        print("Batch of masks shape:", masks.shape)
        # print("Batch of original images shape:", orig_images.shape)
        # print("Batch of original masks shape:", orig_masks.shape)

        # for i in range(images.shape[0]):
        #     print(f"Saving {i} from batch")
            # np.save(f"/results/transformed_{i}.npy", images[i, 0, ...])
            # np.save(f"/results/masks_{i}.npy", masks[i, 0, ...])
            # np.save(f"/results/orig_images_{i}.npy", orig_images[i, 0, ...])
            # np.save(f"/results/orig_masks_{i}.npy", orig_masks[i, 0, ...])
        # break

# Example usage
if __name__ == "__main__":
    main()

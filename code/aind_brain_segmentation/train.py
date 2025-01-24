import os

import lightning as L
import numpy as np
import torchio as tio

import wandb
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from pathlib import Path

import wandb
from dataloader import ImageMaskDataset
from model.network import Neuratt
# from monai.transforms import (
#     Compose,
#     RandFlip,
#     RandAffine,
#     RandGridDistortion,
#     ScaleIntensityRange,
#     NormalizeIntensity
# )
from sklearn.model_selection import KFold
import torch
from pathlib import Path
from torch.utils.data import Subset
import numpy as np
import random
import torch
import warnings
warnings.filterwarnings("ignore")

def random_intensity_scaling(data, scale_range=(0.8, 1.2)):
    """
    Randomly scale the intensity of 3D data.

    Parameters:
        data (numpy.ndarray): The 3D data array to augment.
        scale_range (tuple): The range of scaling factors (min, max).

    Returns:
        numpy.ndarray: The augmented 3D data with adjusted intensities.
    """
    scale_factor = np.random.uniform(*scale_range)
    return data * scale_factor

def random_intensity_shift(data, shift_range=(-0.1, 0.1)):
    """
    Randomly shift the intensity of 3D data.

    Parameters:
        data (numpy.ndarray): The 3D data array to augment.
        shift_range (tuple): The range of intensity shifts (min, max).

    Returns:
        numpy.ndarray: The augmented 3D data with adjusted intensities.
    """
    shift = np.random.uniform(*shift_range)
    return data + shift

class RandomIntensityTransform:
    """
    Custom transform to adjust intensity for 16-bit 3D images with a probability.
    """
    def __init__(self, scale_range=(0.8, 1.2), shift_range=(-1000, 1000), clip_range=(0, 65535), p=0.5):
        """
        Parameters:
            scale_range (tuple): Range for intensity scaling factors.
            shift_range (tuple): Range for intensity shifts.
            clip_range (tuple): Min and max values for clipping (e.g., 0-65535 for 16-bit).
            p (float): Probability of applying the transformation.
        """
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.clip_range = clip_range
        self.p = p

    def __call__(self, tensor):
        if random.random() < self.p:
            scale_factor = random.uniform(*self.scale_range)
            tensor = tensor * scale_factor
            
            shift = random.uniform(*self.shift_range)
            tensor = tensor + shift
            
            tensor = torch.clamp(torch.from_numpy(tensor), min=self.clip_range[0], max=self.clip_range[1])
        return tensor

def get_transforms():
    shift_range = 45
    geometric_prob = 0.5
    intensity_prob = 0.5
    geometric_transform = tio.OneOf({
        tio.RandomFlip(
            axes=(0, 1, 2),
            p=geometric_prob,
            copy=False,
        ),
        tio.RandomAffine(scales=(0.7, 0.5, 0.5), degrees=10, isotropic=False, p=geometric_prob, copy=False),
        tio.RandomElasticDeformation(num_control_points=7, max_displacement=5.0, p=geometric_prob, copy=False),
    })
    intensity_transform = tio.OneOf({
        tio.RandomGamma(log_gamma=(-0.3, 0.3), p=intensity_prob),
        # RandomIntensityTransform(scale_range=(0.3, 2.5), shift_range=(-shift_range, shift_range), p=intensity_prob)
    })
    return tio.Compose([
        geometric_transform,
        intensity_transform,
        tio.RandomBlur(std=(1, 4), p=0.5, copy=False),
        tio.RandomBiasField(coefficients=(0.1, 0.3), order=3, p=0.5, copy=False),
        tio.ZNormalization(copy=False),
        # tio.RescaleIntensity(out_min_max=(0, 1)),  # Normalize intensity values
    ])


# def get_transforms_monai():
#     return Compose([
#         # Randomly flip along x, y, z
#         RandFlip(spatial_axis=(0, 1, 2), prob=0.5),

#         # Random affine transformation
#         RandAffine(
#             rotate_range=(10, 10, 10),  # Rotation in degrees
#             scale_range=(0.1, 0.1, 0.1),  # Scaling factors
#             prob=0.5
#         ),

#         # Random elastic deformation using grid distortion
#         RandGridDistortion(
#             num_cells=(7, 7, 7),  # Control grid size
#             distort_limit=0.5,  # Maximum displacement
#             prob=0.5
#         ),

#         # Normalize intensity values
#         NormalizeIntensity(),

#         # Optionally rescale intensity values to a specific range
#         # ScaleIntensityRange(a_min=0, a_max=1, b_min=0, b_max=1)
#     ])

def train_with_kfold(
    data_path: str,
    logger: L.pytorch.loggers,
    batch_size: int,
    num_workers: int = 0,
    n_splits: int = 5,
    checkpoint_path: str = None,
):
    """
    Train with K-Fold Cross-Validation

    Parameters
    ----------
    data_path: str
        Path where the dataset is located.

    logger: L.pytorch.Loggers
        Logger object.

    batch_size: int
        Batch size for images.

    num_workers: int
        Number of workers in the data loader.

    n_splits: int
        Number of folds for cross-validation.
    """
    seed_everything(42, workers=True)
    results_folder = os.path.abspath("/results")
    
    data_path = Path(data_path)
    dataset = ImageMaskDataset(
        data_path.joinpath('images'),
        data_path.joinpath('masks'),
        transform=get_transforms(),
    )
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
        print(f"Starting Fold {fold + 1}/{n_splits}")
        
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        train_dataloader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        
        val_dataloader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        # Model
        segmentation_model = Neuratt()

        # Checkpoints
        model_checkpoint = ModelCheckpoint(
            monitor="validation/metrics/dice_score",
            filename=f"fold_{fold + 1}_best_model",
            save_top_k=1,
            mode="max",
            verbose=True,
        )
        
        callbacks = [model_checkpoint]

        # Trainer
        trainer = L.Trainer(
            default_root_dir=results_folder,
            callbacks=callbacks,
            logger=logger,
            max_epochs=100,
            max_time="00:04:00:00",
            devices=1,
            accelerator="gpu",
            log_every_n_steps=10,
        )

        # Train the model
        trainer.fit(
            model=segmentation_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        val_metrics = trainer.callback_metrics
        print(f"Fold {fold + 1} Results: {val_metrics}")
        fold_results.append(val_metrics)

    # Summarize results across folds
    print("Cross-Validation Results:")
    for i, metrics in enumerate(fold_results):
        print(f"Fold {i + 1}: {metrics}")

def train(
    train_path: str,
    validation_path: str,
    logger: L.pytorch.loggers,
    batch_size: int,
    num_workers: int = 0,
    checkpoint_path: str = None
):
    """
    Train function

    Parameters
    ----------
    train_path: str
        Path where the training data is located

    validation_path: str
        Path where the validation data is located

    logger: L.pytorch.Loggers
        Logger object

    batch_size: int
        Batch size for images

    num_workers: int
        Number of workers in the data loader.
        Default: 0 (main worker)

    """
    seed_everything(42, workers=True)
    results_folder = os.path.abspath("/results")

    monai_transforms = get_transforms() # get_transforms_monai()
    # Train and validation datasets

    train_path = Path(train_path)

    dataset = ImageMaskDataset(
        train_path.joinpath('images'),
        train_path.joinpath('masks'),
        transform=monai_transforms
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    validation_path = Path(validation_path)
    
    val_dataset = ImageMaskDataset(
        validation_path.joinpath('images'),
        validation_path.joinpath('masks'),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    
    # for images, masks in dataloader:
    #     print("Batch of images shape:", images.shape)
    #     print("Batch of masks shape:", masks.shape)
    #     # print("Batch of original images shape:", orig_images.shape)
    #     # print("Batch of original masks shape:", orig_masks.shape)
    #     break
    # exit()
    # Model
    segmentation_model = Neuratt()

    if checkpoint_path:
        print(f"Loading path from {checkpoint_path}")
        segmentation_model = Neuratt.load_from_checkpoint(checkpoint_path)

    # Checkpoints
    model_checkpoint = ModelCheckpoint(
        monitor="validation/metrics/dice_score", # val_loss
        filename="best_model",
        save_top_k=1,
        mode="max", # min
        verbose=True,
    )

    callbacks = [model_checkpoint]

    # Logging gradients
    wandb.watch(
        segmentation_model,
        log='gradients',
        log_freq=100,
    )
    
    # Trainer
    trainer = L.Trainer(
        default_root_dir=results_folder,
        callbacks=callbacks,
        logger=logger,
        max_epochs=100,
        max_time="00:08:00:00",
        devices=1,
        accelerator="gpu",
        # deterministic=True,
        # overfit_batches=1,
        log_every_n_steps=10,
        limit_predict_batches=10,
    )
    trainer.fit(
        model=segmentation_model,
        train_dataloaders=dataloader,
        val_dataloaders=val_dataloader,
    )

    predictions = trainer.predict(segmentation_model, val_dataloader)

    for i, (data, pred, metrics) in enumerate(predictions):
        np.save(file=f"{results_folder}/data_{i}.npy", arr=data)
        np.save(file=f"{results_folder}/pred_{i}.npy", arr=pred)


if __name__ == "__main__":
    # logger = CSVLogger("/results/whole_brain_seg", name="model-01")
    # previous_run_id = "your_run_id_here"
    project = "whole_brain_seg"
    name = "dice-focal-clamp-intensity"

    run = wandb.init(
        project=project,
        name=name,
    )

    logger = WandbLogger(
        project="whole_brain_seg",  # Replace with your W&B project name
        name="model_dice_focal_clamp_intensity",  # Replace with your specific experiment name
        save_dir="/results/whole_brain_seg",  # Local directory to save logs
        # id=previous_run_id,  # Use the previous run ID
        # resume="allow",
    )
    train(
        train_path="/scratch/dataset_patch_128_steps_64/train",
        validation_path="/scratch/dataset_patch_128_steps_64/test",
        logger=logger,
        batch_size=8,
        num_workers=8,
        checkpoint_path=None
    )
    # "/results/whole_brain_seg/whole_brain_seg/0fp8w3op/checkpoints/best_model.ckpt"
    # train_with_kfold(
    #     data_path="/scratch/dataset_patch_128_steps_64/train",
    #     logger=logger,
    #     batch_size=8,
    #     num_workers=8,
    #     n_splits=5,  # Number of folds
    # )

import logging
import multiprocessing
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
# import cupy as cp
import torchvision.transforms.functional as TF
import zarr
from aind_large_scale_prediction.generator.dataset import create_data_loader
from aind_large_scale_prediction.generator.utils import (
    concatenate_lazy_data, recover_global_position, unpad_global_coords)
from aind_large_scale_prediction.io import ImageReaderFactory
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter
from skimage.morphology import ball, remove_small_objects
from skimage.transform import resize as ski_resize

from aind_brain_segmentation.model.network import Neuratt


def post_process_mask(mask, threshold=0.5, min_size=100):
    mask = binary_fill_holes(mask)
    mask = remove_small_objects(mask, min_size=min_size)

    # structuring_element = ball(radius=5)
    mask = binary_closing(mask)  # , structure=structuring_element)

    mask = gaussian_filter(mask.astype(float), sigma=1)

    mask = mask > threshold
    return mask

def in_mem_computation(
    lazy_data,
    segmentation_model,
    output_seg_path,
    output_prob_path,
    output_data_path,
    target_size_mb,
    n_workers,
    batch_size,
    super_chunksize,
):
    pass

def lazy_computation(
    lazy_data,
    segmentation_model,
    output_seg_path,
    output_prob_path,
    output_data_path,
    target_size_mb,
    n_workers,
    batch_size,
    super_chunksize,
):
    # Lazy computation code
    device = None

    pin_memory = True
    if device is not None:
        pin_memory = False
        multiprocessing.set_start_method("spawn", force=True)

    axis_pad = 0
    overlap_prediction_chunksize = (axis_pad, axis_pad, axis_pad)

    prediction_chunksize = (4, lazy_data.shape[-2], lazy_data.shape[-1])

    logger = logging.Logger(name="log")

    print("Loaded lazy data: ", lazy_data)
    batch_size = 1
    dtype = np.float32
    zarr_data_loader, zarr_dataset = create_data_loader(
        lazy_data=lazy_data,
        target_size_mb=target_size_mb,
        prediction_chunksize=prediction_chunksize,
        overlap_prediction_chunksize=overlap_prediction_chunksize,
        n_workers=n_workers,
        batch_size=batch_size,
        dtype=dtype,  # Allowed data type to process with pytorch cuda
        super_chunksize=super_chunksize,
        lazy_callback_fn=None,  # partial_lazy_deskewing,
        logger=logger,
        device=device,
        pin_memory=pin_memory,
        override_suggested_cpus=False,
        drop_last=True,
        locked_array=False,
    )

    total_batches = sum(zarr_dataset.internal_slice_sum) / batch_size
    print("Total batches: ", total_batches, zarr_dataset.lazy_data.shape)

    # Creating outputs
    output_intermediate_seg = zarr.open(
        output_seg_path,
        "w",
        shape=(
            1,
            1,
        )
        + zarr_dataset.lazy_data.shape[-3:],
        chunks=(
            1,
            1,
        )
        + (128, 128, 128),
        dtype=np.uint8,
    )

    output_intermediate_prob = zarr.open(
        output_prob_path,
        "w",
        shape=(
            1,
            1,
        )
        + zarr_dataset.lazy_data.shape[-3:],
        chunks=(
            1,
            1,
        )
        + (128, 128, 128),
        dtype=np.float16,
    )

    output_raw_data = None

    # Saving raw data if needed
    if output_data_path is not None:
        output_raw_data = zarr.open(
            output_data_path,
            "w",
            shape=(
                1,
                1,
            )
            + zarr_dataset.lazy_data.shape[-3:],
            chunks=(
                1,
                1,
            )
            + (128, 128, 128),
            dtype=np.uint16,
        )

    shape = zarr_dataset.lazy_data.shape[-3:]
    segmentation_model.eval()
    cuda_device = torch.device(0)
    orig_shape = lazy_data.shape[-2:]

    for i, sample in enumerate(zarr_data_loader):
        # Load batch as a NumPy array (but keep it on GPU when possible)
        slice_data_orig = np.squeeze(sample.batch_tensor.numpy())

        # Preallocate array in GPU memory for batch resizing
        slice_data = torch.zeros(
            (slice_data_orig.shape[0], 1, image_height, image_width),
            device=cuda_device,
            dtype=torch.float32,
        )

        # print("Slice orig:", slice_data_orig.shape)

        # **GPU-Accelerated Resizing**
        for i in range(slice_data_orig.shape[0]):
            slice_img = (
                torch.tensor(slice_data_orig[i], device=cuda_device)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            slice_data[i] = F.interpolate(
                slice_img,
                size=(image_height, image_width),
                mode="bilinear",
                align_corners=False,
            )

        # print("Slice data:", slice_data.shape)

        # **Model Prediction**
        pred_mask, prob_mask = segmentation_model.predict(
            batch=slice_data, threshold=prob_threshold
        )

        # print(f"Block {slice_data.shape} - Max mask: {prob_mask.max()}")

        # Move to CPU and convert to NumPy in one step (avoiding unnecessary copies)
        pred_mask = pred_mask.squeeze().detach().cpu().numpy()
        prob_mask = prob_mask.squeeze().detach().cpu().numpy()

        # **Preallocate Resized Masks on CPU**
        pred_mask_resampled = np.zeros(
            (pred_mask.shape[0],) + orig_shape, dtype=np.uint8
        )
        prob_mask_resampled = np.zeros(
            (prob_mask.shape[0],) + orig_shape, dtype=np.float32
        )

        # **Resize with OpenCV (Faster than skimage)**
        for i in range(pred_mask.shape[0]):
            pred_mask_resampled[i] = cv2.resize(
                pred_mask[i], orig_shape[::-1], interpolation=cv2.INTER_NEAREST
            )
            prob_mask_resampled[i] = cv2.resize(
                prob_mask[i], orig_shape[::-1], interpolation=cv2.INTER_LINEAR
            )

        # **Compute Global Positions Efficiently**
        (
            global_coord_pos,
            global_coord_positions_start,
            global_coord_positions_end,
        ) = recover_global_position(
            super_chunk_slice=sample.batch_super_chunk[0],
            internal_slices=sample.batch_internal_slice,
        )

        # **Get global coords position after unpadding**
        # it does not do anything if overlap is 0
        unpadded_global_slice, unpadded_local_slice = unpad_global_coords(
            global_coord_pos=global_coord_pos[-3:],
            block_shape=slice_data.shape[-3:],
            overlap_prediction_chunksize=overlap_prediction_chunksize[-3:],
            dataset_shape=zarr_dataset.lazy_data.shape[-3:],
        )

        # **Move Tensors Off GPU to Free Memory**
        del slice_data
        torch.cuda.empty_cache()

        unpadded_global_slice = (slice(0, 1), slice(0, 1)) + unpadded_global_slice

        print(
            f"Tensor shape: {sample.batch_tensor.shape} - Pred mask -> {pred_mask.shape} - unpadded_global_slice: {unpadded_global_slice} Max mask: {prob_mask.max()}"
        )
        # **Store Output Efficiently**
        output_intermediate_seg[unpadded_global_slice] = pred_mask_resampled[
            None, None, ...
        ]
        output_intermediate_prob[unpadded_global_slice] = prob_mask_resampled[
            None, None, ...
        ]

        if output_data_path is not None:
            output_raw_data[unpadded_global_slice] = slice_data_orig[None, None, ...]

def run_brain_segmentation(
    image_path,
    model_path,
    output_folder,
    target_size_mb=2048,
    n_workers=0,
    super_chunksize=None,
    scale=3,
    scratch_folder=None,
    image_height=1024,
    image_width=1024,
    prob_threshold=0.7,
    run_in_mem=True
):
    output_folder = Path(output_folder)

    if not output_folder.exists():
        raise ValueError(f"Please, provide a valid output path. Path: {output_folder}")

    model_path = Path(model_path)

    # Creating model
    segmentation_model = Neuratt()

    if model_path.exists():
        print(f"Loading path from {model_path}")
        segmentation_model = Neuratt.load_from_checkpoint(str(model_path))
    else:
        raise ValueError(f"Please, provide a valid model path")

    output_seg_path = output_folder.joinpath("segmentation_mask.zarr")
    output_prob_path = output_folder.joinpath("probabilities.zarr")
    output_data_path = None

    if scratch_folder is not None:
        scratch_folder = Path(scratch_folder)
        output_data_path = scratch_folder.joinpath("data.zarr")

    lazy_data = (
        ImageReaderFactory()
        .create(data_path=str(image_path), parse_path=False, multiscale=scale)
        .as_dask_array()
    )

    if run_in_mem:
        in_mem_computation(
            lazy_data,
            segmentation_model,
            output_seg_path,
            output_prob_path,
            output_data_path,
            target_size_mb,
            n_workers,
            batch_size,
            super_chunksize,
        )

    else:
        # Running lazily if dataset is too big
        lazy_computation(
            lazy_data,
            segmentation_model,
            output_seg_path,
            output_prob_path,
            output_data_path,
            target_size_mb,
            n_workers,
            batch_size,
            super_chunksize,
        )

    print("Segmentation finished!")


def main():
    data_folder = Path(os.path.abspath("../data"))
    results_folder = Path(os.path.abspath("../results"))
    scratch_folder = Path(os.path.abspath("../scratch"))

    image_path = "s3://aind-open-data/SmartSPIM_764220_2025-01-30_11-15-58_stitched_2025-03-06_10-04-25/image_tile_fusing/OMEZarr/Ex_639_Em_680.zarr"
    model_path = data_folder.joinpath("best_model_097_2d.ckpt")

    run_brain_segmentation(
        image_path=image_path,
        model_path=model_path,
        output_folder=results_folder,
        target_size_mb=2048,
        n_workers=0,
        super_chunksize=None,
        scale=3,
        scratch_folder=None,
        image_height=1024,
        image_width=1024,
        prob_threshold=0.7,
    )


if __name__ == "__main__":
    main()

"""
Large-scale prediction of segmentation masks.
"""

import logging
import multiprocessing
import os
import re
import time
from pathlib import Path
from typing import Optional
import argparse

import cv2
import dask.array as da
import numpy as np
import torch
import torch.nn.functional as F
import zarr
from aind_brain_segmentation.model.network import Neuratt
from aind_large_scale_prediction.generator.dataset import create_data_loader
from aind_large_scale_prediction.generator.utils import (
    recover_global_position, unpad_global_coords)
from aind_large_scale_prediction.io import ImageReaderFactory
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter
from skimage.morphology import remove_small_objects


def post_process_mask(
    mask: np.ndarray, threshold: Optional[float] = 0.5, min_size: Optional[int] = 100
):
    """
    Post-process segmentation mask.

    Parameters
    ----------
    mask: np.ndarray
        Mask to be postprocessed.

    threshold: Optional[float]
        Threshold for probabilities.

    min_size: Optional[int]
        Minimum size for removing small objects

    """
    mask = binary_fill_holes(mask)
    mask = remove_small_objects(mask, min_size=min_size)

    # structuring_element = ball(radius=5)
    mask = binary_closing(mask)  # , structure=structuring_element)

    mask = gaussian_filter(mask.astype(float), sigma=1)

    mask = mask > threshold
    return mask


def check_gpu_memory(
    image: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: Optional[int] = 1,
    factor: Optional[float] = 2.5,
):
    """
    Checks if both the image and model fit in GPU memory.

    Parameters
    ----------
    image: np.ndarray
        The 3D image array.

    model: torch.nn.Module
        The deep learning model.

    device: torch.device
        Device to use in the prediction

    batch_size: Optional[int]
        The number of images per batch.
        Default: 1

    factor: Optional[float]
        Factor used for memory estimation as
        an overhead. Default: 2.5

    Raises:
        RuntimeError: If the model and image do not fit in GPU memory.
    """
    # Get total and available GPU memory
    total_mem = torch.cuda.get_device_properties(device).total_memory
    reserved_mem = torch.cuda.memory_reserved(device)
    allocated_mem = torch.cuda.memory_allocated(device)
    available_mem = total_mem - max(reserved_mem, allocated_mem)

    # Estimate model memory usage
    model_size = sum(p.numel() * p.element_size() for p in model.parameters())

    # Estimate image memory usage
    image_size = image.nbytes * batch_size

    # Additional overhead: activations, gradients (rough estimate: 2-3x model size)
    overhead = model_size * factor  # Adjust multiplier if needed

    # Total memory requirement
    print(f"Image size: {image_size / (1024 ** 2)} MB")
    print(f"Model size: {model_size / (1024 ** 2)} MB")
    print(f"Computation overhead: {overhead / (1024 ** 2)} MB")

    required_mem = model_size + image_size + overhead

    # Check if it fits
    if required_mem > available_mem:
        raise RuntimeError(
            f"Not enough GPU memory! Required: {required_mem / 1e6:.2f} MB, Available: {available_mem / 1e6:.2f} MB"
        )

    print(
        f"Memory check passed ✅: Required {required_mem / 1e6:.2f} MB, Available {available_mem / 1e6:.2f} MB"
    )


def batched_predict(
    segmentation_model: torch.nn.Module,
    slice_data: np.ndarray,
    prob_threshold: float,
    batch_size: int,
):
    """
    Batched prediction with the segmentation model.

    Parameters
    ----------
    segmentation_model: torch.nn.Module
        Segmentation model

    slice_data: np.ndarray
        Slice data to predict the mask

    prob_threshold: float
        Threshold to cut the probabilities.

    batch_size: int
        Batch size

    """
    if isinstance(slice_data, np.ndarray):
        slice_data = torch.from_numpy(slice_data)

    num_slices = slice_data.shape[0]
    pred_masks, prob_masks = [], []

    for i in range(0, num_slices, batch_size):
        batch = slice_data[i : i + batch_size]
        print(f"Processing slices [{i} - {i+batch_size}]: {batch.shape}")
        pred_mask, prob_mask = segmentation_model.predict(
            batch=batch, threshold=prob_threshold
        )

        pred_masks.append(pred_mask.detach().cpu().numpy())
        prob_masks.append(prob_mask.detach().cpu().numpy())

    # Concatenate results across batch dimension
    pred_masks = np.concatenate(pred_masks, axis=0)
    prob_masks = np.concatenate(prob_masks, axis=0)

    return pred_masks, prob_masks


def in_mem_computation(
    lazy_data,
    segmentation_model,
    output_seg_path,
    output_prob_path,
    output_data_path,
    image_height,
    image_width,
    prob_threshold,
    inner_batch_size=1,
):

    # Creating outputs
    output_intermediate_seg = zarr.open(
        output_seg_path,
        "w",
        shape=(
            1,
            1,
        )
        + lazy_data.shape[-3:],
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
        + lazy_data.shape[-3:],
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
            + lazy_data.shape[-3:],
            chunks=(
                1,
                1,
            )
            + (128, 128, 128),
            dtype=np.uint16,
        )

    shape = lazy_data.shape[-3:]
    cuda_device = torch.device(0)
    segmentation_model.half()
    segmentation_model.eval()
    segmentation_model.to(cuda_device)

    orig_shape = lazy_data.shape[-2:]

    total_mem_gpu = torch.cuda.get_device_properties(cuda_device).total_memory

    # Moving data to GPU if it fits
    print(f"Memory allocated in GPU: {torch.cuda.memory_allocated() / (1024 ** 2)} MB")
    print(f"Memory reserved in GPU: {torch.cuda.memory_reserved() / (1024 ** 2)} MB")
    print(f"Total memory in GPU: {total_mem_gpu / (1024 ** 2)} MB")

    lazy_data = da.squeeze(lazy_data)
    in_mem_data = lazy_data.compute()
    in_mem_data = np.expand_dims(in_mem_data, axis=1).astype(np.float32)

    check_gpu_memory(
        image=in_mem_data,
        model=segmentation_model,
        device=cuda_device,
        batch_size=1,
        factor=2.5,
    )

    gpu_mem_data = torch.from_numpy(in_mem_data).half().to(cuda_device)

    slice_data = torch.zeros(
        (gpu_mem_data.shape[0], 1, image_height, image_width),
        device=cuda_device,
        dtype=gpu_mem_data.dtype,
    )

    for i in range(slice_data.shape[0]):
        slice_data[i] = F.interpolate(
            gpu_mem_data[i][None, ...],
            size=(image_height, image_width),
            mode="bilinear",
            align_corners=False,
        )[0]

    # Removing from GPU, keeping resized
    del gpu_mem_data

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    print(f"Running segmentation in volume of shape: {slice_data.shape}")
    # pred_mask, prob_mask = segmentation_model.predict(
    #     batch=slice_data, threshold=prob_threshold
    # )
    pred_mask, prob_mask = batched_predict(
        segmentation_model, slice_data, prob_threshold, inner_batch_size
    )

    # If we're running multiple datasets at the same time
    del slice_data
    del segmentation_model

    pred_mask = np.squeeze(pred_mask).astype(np.float32)
    prob_mask = np.squeeze(prob_mask)

    print(
        f"After running segmentation, pred mask: {pred_mask.shape} - Prob mask: {prob_mask.shape}"
    )

    if np.isnan(prob_mask).any() or np.isnan(pred_mask).any():
        raise ValueError("NaNs during prediction!")

    if np.isinf(prob_mask).any() or np.isinf(pred_mask).any():
        raise ValueError("INF during prediction!")

    pred_mask_resampled = np.zeros((pred_mask.shape[0],) + orig_shape, dtype=np.uint8)
    prob_mask_resampled = np.zeros((prob_mask.shape[0],) + orig_shape, dtype=np.float32)

    # **Resize with OpenCV (Faster than skimage)**
    for i in range(pred_mask.shape[0]):
        if prob_mask[i] is None or prob_mask[i].size == 0:
            raise ValueError(
                f"Invalid input at index {i}: prob_mask[i] is None or empty."
            )

        if pred_mask[i] is None or pred_mask[i].size == 0:
            raise ValueError(
                f"Invalid input at index {i}: prob_mask[i] is None or empty."
            )

        pred_mask_resampled[i] = cv2.resize(
            pred_mask[i], orig_shape[::-1], interpolation=cv2.INTER_NEAREST
        )
        prob_mask_resampled[i] = cv2.resize(
            prob_mask[i], orig_shape[::-1], interpolation=cv2.INTER_NEAREST
        )

    print(
        f"After resizing segmentation, pred mask: {pred_mask_resampled.shape} - Prob mask: {prob_mask_resampled.shape}"
    )

    print("Writing outputs!")
    output_intermediate_seg[:] = pred_mask_resampled[None, None, ...]
    output_intermediate_prob[:] = prob_mask_resampled[None, None, ...]

    if output_data_path is not None:
        in_mem_data = np.squeeze(in_mem_data)
        output_raw_data[:] = in_mem_data[None, None, ...]


def lazy_computation(
    lazy_data,
    segmentation_model: torch.nn.Module,
    output_seg_path: str,
    output_prob_path: str,
    output_data_path: str,
    target_size_mb: int,
    n_workers: int,
    batch_size: int,
    super_chunksize: int,
    image_height: int,
    image_width: int,
    prob_threshold: float,
):
    """
    Runs whole brain segmentation using 2D planes
    of the data. The orientation of the brain it
    is agnostic since it was trained from multiple
    planes. This is by using lazy data and computation.

    Parameters
    ----------
    lazy_data: dask.Array
        Lazy data

    segmentation_model: torch.nn.Module
        Segmentation model to use for prediction

    output_seg_path: str
        Path where we want to write the zarr for the
        segmentation mask.

    output_prob_path: str
        Path where we want to write the zarr for the
        probabilities.

    output_data_path: str
        Path where we want to write the zarr for the
        raw data.

    n_workers: int
        Number of workers that will be pulling data
        with pytorch dataloaders.

    batch_size: int
        Batch size used during prediction.

    super_chunksize: int
        Superchunksize for large-scale predictions for
        Datasets that are not able to fit in memory.
        This is not allowed by default since for the
        model we need to resize the data.
        Default: None

    scale: int
        Scale from which we will pull the image data in
        SmartSPIM datasets.

    scratch_folder: str
        Scratch folder. If a scratch folder is provided,
        the raw data will be written there for analysis.
        Default: None

    image_height: int
        Image height that will be used for resizing. This size
        was used for the model during training.

    image_width: int
        Image width that will be used for resizing. This size
        was used for the model during training.

    prob_threshold: float
        Probability threshold.

    """

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

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run_brain_segmentation(
    image_path: str,
    model_path: str,
    output_folder: str,
    target_size_mb: Optional[int] = 2048,
    n_workers: Optional[int] = 0,
    super_chunksize: Optional[int] = None,
    scale: Optional[int] = 3,
    scratch_folder: Optional[str] = None,
    image_height: Optional[int] = 1024,
    image_width: Optional[int] = 1024,
    prob_threshold: Optional[float] = 0.7,
    batch_size: Optional[int] = 1,
    run_in_mem: Optional[bool] = True,
):
    """
    Runs whole brain segmentation using 2D planes
    of the data. The orientation of the brain it
    is agnostic since it was trained from multiple
    planes.

    Parameters
    ----------
    image_path: str
        Path where the data is stored. It could be
        a local path or a s3:// path.

    model_path: str
        Path where the model is stored.

    output_folder: str
        Path where we want to write the results.

    n_workers: Optional[int]
        Number of workers that will be pulling data
        with pytorch dataloaders.

    super_chunksize: Optional[int]
        Superchunksize for large-scale predictions for
        Datasets that are not able to fit in memory.
        This is not allowed by default since for the
        model we need to resize the data.
        Default: None

    scale: Optional[int]
        Scale from which we will pull the image data in
        SmartSPIM datasets.

    scratch_folder: str
        Scratch folder. If a scratch folder is provided,
        the raw data will be written there for analysis.
        Default: None

    image_height: Optional[int]
        Image height that will be used for resizing. This size
        was used for the model during training.
        Default: 1024

    image_width: Optional[int]
        Image width that will be used for resizing. This size
        was used for the model during training.
        Default: 1024

    prob_threshold: Optional[float]
        Probability threshold.
        Default: 0.7

    batch_size: Optional[int]
        Batch size used during prediction.

    run_in_mem: Optional[bool]
        Boolean that dictates if the algorithm runs directly in
        memory. False if dask needs to be used.
        Default: True
    """
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
    # lazy_data = da.squeeze(lazy_data)

    if run_in_mem:
        in_mem_computation(
            lazy_data,
            segmentation_model,
            output_seg_path,
            output_prob_path,
            output_data_path,
            image_height=image_height,
            image_width=image_width,
            prob_threshold=prob_threshold,
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
            super_chunksize,
            image_height=image_height,
            image_width=image_width,
            prob_threshold=prob_threshold,
            batch_size=batch_size,
        )

    print("Segmentation finished!")


def run_multiple_datasets():
    """
    Runs segmentation in multiple datasets
    """
    data_folder = Path(os.path.abspath("../data"))
    results_folder = Path(os.path.abspath("../results"))
    # scratch_folder = Path(os.path.abspath("../scratch"))

    image_paths = [
        # "s3://aind-open-data/SmartSPIM_774928_2024-12-17_17-41-54_stitched_2025-01-11_01-02-44/image_tile_fusing/OMEZarr/Ex_639_Em_667.zarr",
        # "s3://aind-open-data/SmartSPIM_764220_2025-01-30_11-15-58_stitched_2025-03-06_10-04-25/image_tile_fusing/OMEZarr/Ex_639_Em_680.zarr",
        # "s3://aind-open-data/SmartSPIM_782499_2025-03-06_00-01-19_stitched_2025-03-07_05-11-31/image_tile_fusing/OMEZarr/Ex_639_Em_680.zarr",
        # "s3://aind-open-data/SmartSPIM_771602_2025-03-05_22-02-27_stitched_2025-03-07_08-59-06/image_tile_fusing/OMEZarr/Ex_639_Em_667.zarr",
        # "s3://aind-open-data/SmartSPIM_714778_2024-03-12_22-40-29_stitched_2024-03-14_06-09-38/image_tile_fusing/OMEZarr/Ex_639_Em_680.zarr",
        "s3://aind-open-data/SmartSPIM_756457_2024-11-21_19-21-52_stitched_2024-11-23_03-00-56/image_tile_fusing/OMEZarr/Ex_639_Em_667.zarr",
        "s3://aind-open-data/SmartSPIM_754077_2024-11-22_00-05-49_stitched_2024-11-26_19-34-32/image_tile_fusing/OMEZarr/Ex_639_Em_667.zarr",
        "s3://aind-open-data/SmartSPIM_758333_2024-11-21_14-52-06_stitched_2024-11-22_21-41-39/image_tile_fusing/OMEZarr/Ex_639_Em_667.zarr",
        "s3://aind-open-data/SmartSPIM_768498_2025-01-27_15-17-42_stitched_2025-01-29_09-09-04/image_tile_fusing/OMEZarr/Ex_639_Em_680.zarr",
        "s3://aind-open-data/SmartSPIM_768499_2025-01-27_18-45-19_stitched_2025-01-29_08-19-35/image_tile_fusing/OMEZarr/Ex_639_Em_680.zarr",
    ]

    model_path = data_folder.joinpath("smartspim_tissue_segmentation/smartspim_tissue_segmentation.ckpt")

    for image_path in image_paths:
        match = re.search(r"(SmartSPIM_\d+)", image_path)
        smartspim_id = None
        if match:
            smartspim_id = match.group(1)
        else:
            raise ValueError("Please, provide a SmartSPIM ID")

        print(f"Processing dataset: {smartspim_id}")

        curr_res_folder = results_folder.joinpath(smartspim_id)
        curr_res_folder.mkdir(parents=True, exist_ok=True)

        run_brain_segmentation(
            image_path=image_path,
            model_path=model_path,
            output_folder=curr_res_folder,
            target_size_mb=2048,
            n_workers=0,
            super_chunksize=None,
            scale=3,
            scratch_folder=curr_res_folder,
            image_height=1024,
            image_width=1024,
            prob_threshold=0.7,
        )


def main():
    """
    Main function
    """

    parser = argparse.ArgumentParser(description="Run brain segmentation.")

    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--model_path", type=str, default="../data/smartspim_tissue_segmentation/smartspim_tissue_segmentation.ckpt", help="Path to the model checkpoint.")
    parser.add_argument("--output_folder", type=str, default=os.path.abspath("../results"), help="Folder to store results.")
    parser.add_argument("--target_size_mb", type=int, default=2048, help="Target size in MB.")
    parser.add_argument("--n_workers", type=int, default=0, help="Number of workers.")
    parser.add_argument("--super_chunksize", type=int, default=None, help="Super chunk size.")
    parser.add_argument("--scale", type=int, default=3, help="Scale factor.")
    parser.add_argument("--scratch_folder", type=str, default=os.path.abspath("../scratch"), help="Scratch folder.")
    parser.add_argument("--image_height", type=int, default=1024, help="Image height.")
    parser.add_argument("--image_width", type=int, default=1024, help="Image width.")
    parser.add_argument("--prob_threshold", type=float, default=0.7, help="Probability threshold.")

    args = parser.parse_args()

    results_folder = Path(os.path.abspath(args.output_folder))
    scratch_folder = Path(os.path.abspath(args.scratch_folder))

    image_path = args.image_path
    model_path = os.path.abspath(args.model_path)

    run_brain_segmentation(
        image_path=image_path,
        model_path=model_path,
        output_folder=results_folder,
        target_size_mb=2048,
        n_workers=0,
        super_chunksize=None,
        scale=3,
        scratch_folder=scratch_folder,
        image_height=1024,
        image_width=1024,
        prob_threshold=0.7,
    )


if __name__ == "__main__":
    # main()
    run_multiple_datasets()

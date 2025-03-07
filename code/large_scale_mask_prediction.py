import logging
import multiprocessing
import os
import time
from pathlib import Path

import cv2
import dask.array as da
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


def check_gpu_memory(
    image: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 1,
    factor: float = 2.5,
):
    """
    Checks if both the image and model fit in GPU memory.

    Args:
        image (np.ndarray): The 3D image array.
        model (torch.nn.Module): The deep learning model.
        batch_size (int): The number of images per batch.

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


def batched_predict(segmentation_model, slice_data, prob_threshold, batch_size):
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
    target_size_mb,
    n_workers,
    batch_size,
    super_chunksize,
    image_height,
    image_width,
    prob_threshold,
    inner_batch_size=4,
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
    in_mem_data = np.expand_dims(in_mem_data, axis=1)

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
        f"After resizing segmentation, pred mask: {pred_mask.shape} - Prob mask: {prob_mask.shape}"
    )

    print("Writing outputs!")
    output_intermediate_seg[:] = pred_mask_resampled[None, None, ...]
    output_intermediate_prob[:] = prob_mask_resampled[None, None, ...]

    if output_data_path is not None:
        in_mem_data = np.squeeze(in_mem_data)
        output_raw_data[:] = in_mem_data[None, None, ...]


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
    image_height,
    image_width,
    prob_threshold,
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
    run_in_mem=True,
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
    # lazy_data = da.squeeze(lazy_data)

    if run_in_mem:
        in_mem_computation(
            lazy_data,
            segmentation_model,
            output_seg_path,
            output_prob_path,
            output_data_path,
            target_size_mb,
            n_workers,
            1,
            super_chunksize,
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
            batch_size,
            super_chunksize,
            image_height=image_height,
            image_width=image_width,
            prob_threshold=prob_threshold,
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
        scratch_folder=scratch_folder,
        image_height=1024,
        image_width=1024,
        prob_threshold=0.7,
    )


if __name__ == "__main__":
    main()

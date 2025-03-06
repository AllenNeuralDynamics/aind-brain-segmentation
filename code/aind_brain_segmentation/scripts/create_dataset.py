import os
from pathlib import Path
from skimage.measure import regionprops
import tifffile as tif
import re
import numpy as np
from patchify import patchify
from natsort import natsorted
from skimage import io
import warnings
from skimage import exposure, measure
from skimage.transform import resize as ski_resize
import multiprocessing

warnings.filterwarnings('ignore')

def create_folder(dest_dir: str, verbose = False) -> None:
    """
    Create new folders.

    Parameters
    ------------------------

    dest_dir: PathLike
        Path where the folder will be created if it does not exist.

    verbose: Optional[bool]
        If we want to show information about the folder status. Default False.

    Raises
    ------------------------

    OSError:
        if the folder exists.

    """

    if not (os.path.exists(dest_dir)):
        try:
            if verbose:
                print(f"Creating new directory: {dest_dir}")
            os.makedirs(dest_dir)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise

def main(
    dataset_path,
    masks_dataset_path,
    output_path,
    test_brain_ids,
    patch_size=64,
    step_size=64,
    pmin=1,
    pmax=99,
    int_threshold=40,
    apply_percentile_norm=False,
    clip_int=True,
    crop_to_mask=False,
    volume_blocks=True
):
    dataset_path = Path(dataset_path)
    masks_dataset_path = Path(masks_dataset_path)
    regex_id = r"_(\d{6})_"
    pattern = r"Ex_\d{3}_Em_\d{3}"
    
    print("Generating dataset in ", output_path)
    output_path = Path(output_path)

    for fd in ['train', 'test']:
        create_folder(str(output_path.joinpath(f"{fd}/images")))
        create_folder(str(output_path.joinpath(f"{fd}/masks")))

    if dataset_path.exists() and masks_dataset_path.exists():
        masks_paths = list(masks_dataset_path.glob("*_postprocessed.tif*"))
        brain_id_bbox = {}

        if len(masks_paths):
            masks_paths = natsorted(masks_paths)
            
            for mask_path in masks_paths:
                data_block = tif.imread(mask_path)
                print(f"Processing: {mask_path}: {data_block.dtype} - Shape: {data_block.shape}")

                props = regionprops(
                    data_block,
                    intensity_image=None,
                    cache=True,
                    extra_properties=None,
                    spacing=None,
                    offset=None
                )
                largest_region = max(props, key=lambda prop: prop.area)

                match = re.search(regex_id, str(mask_path))

                if match:
                    extracted_number = match.group(1)
                    print(f"SmartSPIM ID: {extracted_number}")
                    brain_id_bbox[extracted_number] = {
                        'bbox': largest_region.bbox,
                        'area': largest_region.area
                    }
                else:
                    raise ValueError("No six-digit number found in path")

                masks_output_path = output_path.joinpath("train/masks")
                images_output_path = output_path.joinpath("train/images")

                if extracted_number in test_brain_ids:
                    masks_output_path = output_path.joinpath("test/masks")
                    images_output_path = output_path.joinpath("test/images")

                # if extracted_number != "729533":
                #     print(f"Skipping {extracted_number}")
                #     continue

                image_paths = list(dataset_path.glob(f"*{extracted_number}*.tif*"))
                print(f"{len(image_paths)} channels for {extracted_number}")
                for image_path in image_paths:
                    # image_block = tif.imread(image_path)

                    matches = re.findall(pattern, str(image_path))
                    if len(matches):
                        channel_name = matches[0]

                    else:
                        channel_name = str(image_path.stem).split("_")[-1]

                    print(f"Channel: {channel_name} for {image_path}")
                    image_block = io.imread(image_path)
    
                    crop_to_mask_region = (
                        slice(None),
                        slice(None),
                        slice(None),
                    )
    
                    if crop_to_mask:
                        region = (
                            slice(largest_region.bbox[0], largest_region.bbox[3]),
                            slice(largest_region.bbox[1], largest_region.bbox[4]),
                            slice(largest_region.bbox[2], largest_region.bbox[5]),
                        )
                        print(f"Cropping to mask in region: {crop_to_mask_region}")
                        
                    extracted_mask_block = data_block[crop_to_mask_region]
                    # [
                    #     largest_region.bbox[0]: largest_region.bbox[3],
                    #     largest_region.bbox[1]: largest_region.bbox[4],
                    #     largest_region.bbox[2]: largest_region.bbox[5],
                    # ]
    
                    extracted_image_block = image_block[crop_to_mask_region]
                    # [
                    #     largest_region.bbox[0]: largest_region.bbox[3],
                    #     largest_region.bbox[1]: largest_region.bbox[4],
                    #     largest_region.bbox[2]: largest_region.bbox[5],
                    # ]
    
                    extracted_mask_block = np.squeeze(extracted_mask_block)
                    extracted_image_block = np.squeeze(extracted_image_block)

                    # if extracted_number != "718357":
                    #     continue
    
                    if extracted_number == "729533":
                        print(f"Fixing orientation of {extracted_number}")
                        
                        extracted_mask_block = np.flip(
                            np.transpose(
                                extracted_mask_block,
                                (-1, 1, 0)
                            ), axis=1
                        )
                        extracted_image_block = np.flip(
                            np.transpose(
                                extracted_image_block,
                                (-1, 1, 0)
                            ), axis=1
                        )
    
                    if extracted_image_block.ndim != 3:
                        raise ValueError(f"Image has the following shape: {extracted_image_block.shape}")
    
                    if extracted_mask_block.ndim != 3:
                        raise ValueError(f"Image has the following shape: {extracted_mask_block.shape}")
    
                    print(f"Writing blocks from {extracted_number} to {images_output_path.parent}")
                    print(f"Image shape: {extracted_image_block.shape} - Mask shape: {extracted_mask_block.shape}")
                    
                    if volume_blocks:
                        patch_image_mask_data(
                            data_block=extracted_image_block,
                            mask_block=extracted_mask_block,
                            output_images=images_output_path,
                            output_masks=masks_output_path,
                            smartspim_id=extracted_number,
                            patch_size=patch_size,
                            step_size=step_size,
                            pmin=pmin,
                            pmax=pmax,
                            int_threshold=int_threshold,
                            apply_percentile_norm=apply_percentile_norm,
                            clip_int=clip_int,
                        )
    
                    else:
                        print("Generating slices!")
                        patch_image_in_slices(
                            data_block=extracted_image_block,
                            mask_block=extracted_mask_block,
                            output_images=images_output_path,
                            output_masks=masks_output_path,
                            smartspim_id=extracted_number,
                            image_width=1024,
                            image_height=1024,
                            pmin=pmin,
                            pmax=pmax,
                            int_threshold=int_threshold,
                            apply_percentile_norm=apply_percentile_norm,
                            clip_int=clip_int,
                            channel_name=channel_name
                        )
                        
                    print("*"*20)
            
        else:
            raise FileNotFoundError(f"Path {dataset_path} does not have tif files.")
        
    else:
        raise FileNotFoundError(f"Path {dataset_path} does not exist")    

def percentile_normalization(data, percentiles=(1, 99), clip=True):
    """
    Normalize data based on percentile values.
    
    Parameters:
    -----------
    data : array-like
        Input data to be normalized
    percentiles : tuple of (float, float), default=(0, 100)
        Tuple containing (lower_percentile, upper_percentile) values
        Values should be in range [0, 100]
    clip : bool, default=True
        If True, clip values outside the percentile range
        
    Returns:
    --------
    normalized_data : numpy.ndarray
        Data normalized to [0, 1] range based on the specified percentiles
        
    Examples:
    --------
    >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> normalized = percentile_normalization(data, percentiles=(10, 90))
    """
    if not isinstance(percentiles, (list, tuple)) or len(percentiles) != 2:
        raise ValueError("percentiles must be a tuple or list of length 2")
    if not (0 <= percentiles[0] < percentiles[1] <= 100):
        raise ValueError("percentiles must be in range [0, 100] and lower < upper")
    
    lower_val, upper_val = np.percentile(data, tuple(percentiles))
    
    print("Max in data: ", np.maximum(data, lower_val), lower_val, upper_val)
    print("Min val in data: ", data.min(), " max val in data: ", data.max())
    data = np.maximum(data, lower_val)
    
    if lower_val == upper_val:
        raise ValueError("Percentile values are identical, division by zero would occur")
    
    if clip:
        data = np.clip(data, lower_val, upper_val)

    normalized_data = (data - lower_val) / (upper_val - lower_val)

    print("After norm Min val in data: ", normalized_data.min(), " max val in data: ", normalized_data.max())
    
    return normalized_data
    
def patch_image_mask_data(
    data_block,
    mask_block,
    output_images,
    output_masks,
    smartspim_id,
    patch_size=64,
    step_size=64,
    pmin=1,
    pmax=99,
    int_threshold=40,
    apply_percentile_norm=False,
    clip_int=True
):
    print("input shapes ", data_block.shape, mask_block.shape)
    output_images = Path(output_images)
    output_masks = Path(output_masks)
    
    padding_needed = [(patch_size - (dim % patch_size)) % patch_size for dim in data_block.shape]
    padding = [(p // 2, p - (p // 2)) for p in padding_needed]
    padded_data_block = np.pad(data_block, padding, mode='constant', constant_values=0)
    padded_mask_block = np.pad(mask_block, padding, mode='constant', constant_values=0)
    
    print("padded shapes ", padded_data_block.shape, padded_mask_block.shape)

    if clip_int:
        print("Applying clipping")
        indices = np.where(padded_data_block <= int_threshold)
        padded_data_block[indices] = 0
        
    if apply_percentile_norm:
        print("Applying percentile norm")
        # print("Min max ", padded_data_block.min(), padded_data_block.max())
        
        padded_data_block = percentile_normalization(
            data=padded_data_block,
            percentiles=(pmin, pmax),
            clip=False
        )

        # print("Padded data block dtype: ", padded_data_block.dtype)

        # io.imsave("/results/check_volume_no_perc.tif", padded_data_block.astype(np.uint16))
        # exit()
    
    patched_image = patchify(
        padded_data_block,
        patch_size=(patch_size, patch_size, patch_size),
        step=(step_size, step_size, step_size)
    ).reshape(-1, patch_size, patch_size, patch_size)

    patched_mask = patchify(
        padded_mask_block,
        patch_size=(patch_size, patch_size, patch_size),
        step=(step_size, step_size, step_size)
    ).reshape(-1, patch_size, patch_size, patch_size)
    
    n_blocks = patched_image.shape[0]

    print(f"Processing {n_blocks} blocks for ID: {smartspim_id}")
    saved_blocks = 0
    
    for i, patched_mask_block in enumerate(patched_mask):
        # if patched_mask_block.max() == 0:
        #     continue

        saved_blocks += 1
        output_image_block = output_images.joinpath(f"{smartspim_id}_image_block_{i}.tif")
        output_mask_block = output_masks.joinpath(f"{smartspim_id}_mask_block_{i}.tif")

        patched_image_block = patched_image[i]
        
        io.imsave(output_image_block, patched_image_block.astype(np.uint16))
        io.imsave(output_mask_block, patched_mask_block.astype(np.uint8))

    print(f"Total blocks: {n_blocks} - Saved blocks: {saved_blocks} - Empty blocks: {n_blocks - saved_blocks}")

def constrat_enhancement(normalized_image, k_factor=8, clip_limit=0.03):
    """
    Contrast enhancement with adaptative histogram equalization.

    Note: The image must be in a range between -1 and 1 for floating images.
    """

    normalized_image_cp = normalized_image.copy()
    kernel_size = (
        normalized_image_cp.shape[0] // k_factor,
        normalized_image_cp.shape[1] // k_factor,
    )

    equalized_image = exposure.equalize_adapthist(
        normalized_image_cp, kernel_size=kernel_size, clip_limit=clip_limit
    )

    return equalized_image

def max_ignore_inf(arr):
    finite_vals = arr[np.isfinite(arr) & ~np.isnan(arr)]  # Mask out infinities
    return np.max(finite_vals) if finite_vals.size > 0 else None

def fix_nans_and_infs(arr):
    min_val = np.nanmin(arr)
    max_val = max_ignore_inf(arr)

    fixed_arr = np.nan_to_num(arr, nan=min_val, posinf=max_val, neginf=min_val)
    
    return fixed_arr

def process_slice(i, axis, data_block, mask_block, output_images, output_masks, smartspim_id, image_width, image_height, channel_name):
    """Process a single slice and save the result."""
    if axis == 0:
        slice_mask = mask_block[i, :, :]
        slice_data = data_block[i, :, :]
    elif axis == 1:
        slice_mask = mask_block[:, i, :]
        slice_data = data_block[:, i, :]
    else:  # axis == 2
        slice_mask = mask_block[:, :, i]
        slice_data = data_block[:, :, i]

    max_id = slice_mask.max()
    max_data_block = slice_data.max()
    # print(f"Job [{os.getpid()}] processing {smartspim_id} plane {i}")

    if max_id and max_data_block:
        slice_data_resized = ski_resize(slice_data, (image_height, image_width), order=4, preserve_range=True)
        slice_mask_resized = ski_resize(slice_mask, (image_height, image_width), order=0, preserve_range=True)

        slice_data_resized = fix_nans_and_infs(slice_data_resized)
        slice_mask_resized = fix_nans_and_infs(slice_mask_resized)

        if np.isnan(slice_data_resized).any() or np.isinf(slice_data_resized).any():
            print(f"[!!!] Problem processing: {smartspim_id}_image_slice_{i}_ax_{axis}")

        if np.isnan(slice_mask_resized).any() or np.isinf(slice_mask_resized).any():
            print(f"[!!!] Problem processing: {smartspim_id}_mask_slice_{i}_ax_{axis}")

        output_image_slice = output_images.joinpath(f"{smartspim_id}_image_slice_{i}_ax_{axis}_{channel_name}.tif")
        output_mask_slice = output_masks.joinpath(f"{smartspim_id}_mask_slice_{i}_ax_{axis}_{channel_name}.tif")

        io.imsave(output_image_slice, slice_data_resized.astype(np.float16))
        io.imsave(output_mask_slice, slice_mask_resized.astype(np.uint8))
    else:
        # print(f"[!] Ignoring slice {i} - Max id: {max_id}")
        pass

def patch_image_in_slices(
    data_block,
    mask_block,
    channel_name,
    output_images,
    output_masks,
    smartspim_id,
    image_width=1024,
    image_height=1024,
    pmin=1,
    pmax=99,
    int_threshold=40,
    apply_percentile_norm=False,
    clip_int=True,
    num_workers=multiprocessing.cpu_count(),
):
    print("input shapes ", data_block.shape, mask_block.shape)
    output_images = Path(output_images)
    output_masks = Path(output_masks)

    if clip_int:
        print("Applying clipping")
        indices = np.where(data_block <= int_threshold)
        data_block[indices] = 0  # Removed `padded_data_block` since it wasn't defined

    if apply_percentile_norm:
        print("Applying percentile norm")
        data_block = percentile_normalization(data=data_block, percentiles=(pmin, pmax), clip=False)

    tasks = []
    for axis in range(3):
        print(f"Processing axis {axis}")
        n_slices = data_block.shape[axis]
        print(f"Processing {n_slices} slices for {smartspim_id} in axis {axis}")

        for i in range(n_slices):
            tasks.append((i, axis, data_block, mask_block, output_images, output_masks, smartspim_id, image_width, image_height, channel_name))

    # Use multiprocessing pool to parallelize
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.starmap(process_slice, tasks)

def patch_image_in_slices_2(
    data_block,
    mask_block,
    output_images,
    output_masks,
    smartspim_id,
    image_width=1024,
    image_height=1024,
    pmin=1,
    pmax=99,
    int_threshold=40,
    apply_percentile_norm=False,
    clip_int=True
):
    print("input shapes ", data_block.shape, mask_block.shape)
    output_images = Path(output_images)
    output_masks = Path(output_masks)

    # patch_size = 64
    # padding_needed = [(patch_size - (dim % patch_size)) % patch_size for dim in data_block.shape]
    # padding = [(p // 2, p - (p // 2)) for p in padding_needed]
    # padded_data_block = np.pad(data_block, padding, mode='constant', constant_values=0)
    # padded_mask_block = np.pad(mask_block, padding, mode='constant', constant_values=0)

    if clip_int:
        print("Applying clipping")
        indices = np.where(data_block <= int_threshold)
        padded_data_block[indices] = 0
        
    if apply_percentile_norm:
        print("Applying percentile norm")
        # print("Min max ", padded_data_block.min(), padded_data_block.max())
        
        padded_data_block = percentile_normalization(
            data=data_block,
            percentiles=(pmin, pmax),
            clip=False
        )

    for axis in range(3):
        n_slices = data_block.shape[0]
        print(f"Processing {axis} with {n_slices} slices for {smartspim_id}")
        for i in range(n_slices):

            slice_mask = None
            slice_data = None
            
            if axis == 0:
                slice_mask = mask_block[i, :, :]
                slice_data = data_block[i, :, :]
                
            elif axis == 1:
                slice_mask = mask_block[:, i, :]
                slice_data = data_block[:, i, :]

            elif axis == 2:
                slice_mask = mask_block[:, :, i]
                slice_data = data_block[:, :, i]
                
            max_id = slice_mask.max()
            max_data_block = slice_data.max()
        
            if max_id and max_data_block:
                # print(f"Processing slice: {i} - counter: {saved_slices}")
                slice_data_resized = ski_resize(
                    slice_data, (image_height, image_width), order=4, preserve_range=True
                )
                slice_mask_resized = ski_resize(
                    slice_mask, (image_height, image_width), order=0, preserve_range=True
                )

                slice_data_resized = fix_nans_and_infs(slice_data_resized)
                slice_mask_resized = fix_nans_and_infs(slice_mask_resized)

                if np.isnan(slice_data_resized).any() or np.isinf(slice_data_resized).any():
                    print(f"Problem processing: {smartspim_id}_image_slice_{i}_ax_{axis}")

                if np.isnan(slice_mask_resized).any() or np.isinf(slice_mask_resized).any():
                    print(f"Problem processing: {smartspim_id}_mask_slice_{i}_ax_{axis}")
                
                output_image_slice = output_images.joinpath(f"{smartspim_id}_image_slice_{i}_ax_{axis}.tif")
                output_mask_slice = output_masks.joinpath(f"{smartspim_id}_mask_slice_{i}_ax_{axis}.tif")
        
                io.imsave(output_image_slice, slice_data_resized.astype(np.float16))
                io.imsave(output_mask_slice, slice_mask_resized.astype(np.uint8))
        
            else:
                # print(f"Ignoring slice {i} - Max id: {max_id}")
                pass
    
if __name__ == "__main__":
    step_sizes = [64]#[64, 128]
    patch_sizes = [64]#[64, 128]
    apply_percentile_norm = False
    clip_int = False
    crop_to_mask = True

    for patch_size in patch_sizes:
        for step_size in step_sizes:
            print(f"Processing with patch size of {patch_size} and step size of {step_size} - Percentiles: {apply_percentile_norm} - clip int {clip_int}")
            output_path = f"/scratch/dataset_patch_{patch_size}_steps_{step_size}"

            if apply_percentile_norm:
                output_path = f"{output_path}_percentile"

            if clip_int:
                output_path = f"{output_path}_clip_int"

            if crop_to_mask:
                output_path = f"{output_path}_croptomask"
            else:
                output_path = f"{output_path}_nocroptomask"
                
            main(
                dataset_path="/scratch/downloaded_dataset_images",
                masks_dataset_path="/data/smartspim_brain_masks",
                output_path=output_path,
                test_brain_ids=['727461', '757189'],
                patch_size=patch_size,
                step_size=step_size,
                pmin=1,
                pmax=99,
                int_threshold=20,
                apply_percentile_norm=apply_percentile_norm,
                clip_int=clip_int,
                crop_to_mask=crop_to_mask,
                volume_blocks=False,
            )

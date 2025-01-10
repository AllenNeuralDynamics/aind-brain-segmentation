import os
from pathlib import Path
from skimage.measure import regionprops
import tifffile as tif
import re
import numpy as np
from patchify import patchify

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
    output_path,
    test_brain_ids,
    patch_size=64,
    step_size=64
):
    dataset_path = Path(dataset_path)
    regex_id = r"_(\d{6})_"

    print("Generating dataset in ", output_path)
    output_path = Path(output_path)

    for fd in ['train', 'test']:
        create_folder(str(output_path.joinpath(f"{fd}/images")))
        create_folder(str(output_path.joinpath(f"{fd}/masks")))

    if dataset_path.exists():
        masks_paths = list(dataset_path.glob("*_postprocessed.tif*"))
        brain_id_bbox = {}

        if len(masks_paths):

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
                
                image_path = list(dataset_path.glob(f"SmartSPIM_{extracted_number}*.tif*"))[0]
                image_block = tif.imread(image_path)
                
                extracted_mask_block = data_block[
                    largest_region.bbox[0]: largest_region.bbox[3],
                    largest_region.bbox[1]: largest_region.bbox[4],
                    largest_region.bbox[2]: largest_region.bbox[5],
                ]

                extracted_image_block = image_block[
                    largest_region.bbox[0]: largest_region.bbox[3],
                    largest_region.bbox[1]: largest_region.bbox[4],
                    largest_region.bbox[2]: largest_region.bbox[5],
                ]

                print(f"Writing blocks from {extracted_number} to {images_output_path.parent}")
                patch_image_mask_data(
                    data_block=extracted_image_block,
                    mask_block=extracted_mask_block,
                    output_images=images_output_path,
                    output_masks=masks_output_path,
                    smartspim_id=extracted_number,
                    patch_size=patch_size,
                    step_size=step_size
                )
                print("*"*20)
            
        else:
            raise FileNotFoundError(f"Path {dataset_path} does not have tif files.")
        
    else:
        raise FileNotFoundError(f"Path {dataset_path} does not exist")    
    
def patch_image_mask_data(
    data_block,
    mask_block,
    output_images,
    output_masks,
    smartspim_id,
    patch_size=64,
    step_size=64
):
    print("input shapes ", data_block.shape, mask_block.shape)
    output_images = Path(output_images)
    output_masks = Path(output_masks)
    
    padding_needed = [(patch_size - (dim % patch_size)) % patch_size for dim in data_block.shape]
    padding = [(p // 2, p - (p // 2)) for p in padding_needed]
    padded_data_block = np.pad(data_block, padding, mode='constant', constant_values=0)
    padded_mask_block = np.pad(mask_block, padding, mode='constant', constant_values=0)
    
    print("padded shapes ", padded_data_block.shape, padded_mask_block.shape)
    
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
        if patched_mask_block.max() == 0:
            continue
        saved_blocks += 1
        output_image_block = output_images.joinpath(f"{smartspim_id}_image_block_{i}.tif")
        output_mask_block = output_masks.joinpath(f"{smartspim_id}_mask_block_{i}.tif")

        patched_image_block = patched_image[i]
        
        tif.imwrite(output_image_block, patched_image_block)
        tif.imwrite(output_mask_block, patched_mask_block)

    print(f"Total blocks: {n_blocks} - Saved blocks: {saved_blocks} - Empty blocks: {n_blocks - saved_blocks}")
    
if __name__ == "__main__":
    step_sizes = [64, 128]
    patch_sizes = [64, 128]

    for patch_size in patch_sizes:
        for step_size in step_sizes:
            print(f"Processing with patch size of {patch_size} and step size of {step_size}")
            main(
                dataset_path="/data/smartspim_brain_masks",
                output_path=f"/scratch/dataset_patch_{patch_size}_steps_{step_size}",
                test_brain_ids=['729674'],
                patch_size=patch_size,
                step_size=step_size
            )

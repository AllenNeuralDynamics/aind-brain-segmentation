import os
from pathlib import Path
from skimage.measure import regionprops
import tifffile as tif
import re

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

def main(dataset_path, output_path):
    dataset_path = Path(dataset_path)

    print("Generating dataset in ", output_path)
    create_folder(output_path)
    output_path = Path(output_path)

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
                
                print("Props bbox: ", largest_region.bbox)

                match = re.search(r"_(\d{6})_", str(mask_path))

                if match:
                    extracted_number = match.group(1)
                    print(f"SmartSPIM ID: {extracted_number}")
                    brain_id_bbox[extracted_number] = {
                        'bbox': largest_region.bbox,
                        'area': largest_region.area
                    }
                else:
                    print("No six-digit number found.")

                # Generating block
                output_block_path = output_path.joinpath(extracted_number)
            
        else:
            raise FileNotFoundError(f"Path {dataset_path} does not have tif files.")
        
    else:
        raise FileNotFoundError(f"Path {dataset_path} does not exist")    
    

if __name__ == "__main__":
    main(
        dataset_path="/data/smartspim_brain_masks"
    )
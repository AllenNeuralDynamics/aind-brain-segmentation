import torch
from pathlib import Path
from safetensors.torch import save_file

def create_safetensors_from_cpkt(
    model_path: str,
    output_path: str
):
    """
    Creates a safetensor from pytorch lightning cpkt.

    Parameters
    ----------
    model_path: str
        Path where the cpkt object is.

    output_path: str
        Path where the .safetensors will be written.

    """
    model_path = Path(model_path)
    output_path = Path(output_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Provide a valid model path: {model_path}")

    if not output_path.exists():
        raise FileNotFoundError(f"Provide a valid model path: {output_path}")
    
    cpkt_model = torch.load(model_path)
    
    state_dict = cpkt_model["state_dict"]

    safetensors_path = output_path.joinpath("model.safetensors")
    print(f"Saving tensors in {safetensors_path}")
    save_file(state_dict, safetensors_path)

if __name__ == "__main__":
    model_path = "/data/smartspim_tissue_segmentation/smartspim_tissue_segmentation.ckpt"
    output_path = "/results"
    create_safetensors_from_cpkt(model_path, output_path)

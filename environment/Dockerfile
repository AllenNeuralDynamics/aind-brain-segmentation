# hash:sha256:a296014577d6b1a62de2425b7fb33bc3f00f3cf2896b034cc3d0be4bab61d0bb
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/pytorch:2.4.0-cuda12.4.0-mambaforge24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update && apt-get install libgl1 -y

RUN conda install -y \
        jupyterlab==4.3.4 \
    && conda clean -ya

RUN pip install --no-cache \
    tifffile==2025.2.18 \
    patchify==0.2.3 \
    scipy==1.15.2 \
    scikit-image==0.25.2 \
    torchio==0.20.4 \
    numpy==1.26.4 \
    lightning==2.5.0.post0 \
    torchmetrics==1.6.2 \
    segmentation_models_pytorch==0.4.0 \
    wandb==0.19.8 \
    torchvision==0.18.1 \
    natsort==8.4.0 \
    dask[distributed]==2024.4.1 \
    git+https://github.com/AllenNeuralDynamics/aind-large-scale-prediction.git@unpinned-versions \
    monai[all]==1.4.0 \
    kornia==0.8.0 \
    boto3==1.37.9 \
    opencv-python==4.11.0

RUN conda install conda-forge::awscli

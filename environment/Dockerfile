# hash:sha256:a296014577d6b1a62de2425b7fb33bc3f00f3cf2896b034cc3d0be4bab61d0bb
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/pytorch:2.4.0-cuda12.4.0-mambaforge24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda install -y \
        jupyterlab==4.3.4 \
    && conda clean -ya

RUN pip install --no-cache \
    git+https://github.com/AllenNeuralDynamics/aind-large-scale-prediction.git@feat-updating-version \
    tifffile \
    patchify \
    scipy \
    scikit-image \
    torchio \
    numpy \
    lightning \
    torchmetrics \
    segmentation_models_pytorch \
    wandb \
    torchvision==0.18.1

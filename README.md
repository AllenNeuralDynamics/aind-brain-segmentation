# aind-brain-segmentation

## Overview
**aind-brain-segmentation** is a tool for large-scale segmentation of brain tissue acquired using the SmartSPIM microscope. The model is trained to segment 2D sagittal, coronal, or axial planes in channel 639.

It supports two processing modes:
1. **Lazy segmentation**: For large datasets that cannot fit into memory, images are resized to **1024x1024**, normalized, segmented, upsampled, and stored in a Zarr format.
2. **In-memory segmentation**: For smaller datasets, a multiscale version is loaded into memory for processing.

The model was trained with a resolution of **14.4 microns in XY** and **16 microns in Z**.

## Input Parameters
| Parameter | Description |
|-----------|-------------|
| `image_path` | Path to the OME-Zarr dataset (local or S3). Example: `s3://bucket/SmartSPIM_Dataset/Ex_639_Em_667.zarr` |
| `model_path` | Path to the trained segmentation model. |
| `output_folder` | Directory where results will be saved. |
| `target_size_mb` | Memory allocation target for large-scale predictions. |
| `n_workers` | Number of workers for PyTorch DataLoader. |
| `super_chunksize` | Data shard size for optimized cloud communication. |
| `scale` | Name of the multiscale dataset to use for segmentation. |
| `scratch_folder` | Path for temporary files. |
| `image_height` | Resize height for segmentation (Default: `1024`). |
| `image_width` | Resize width for segmentation (Default: `1024`). |
| `prob_threshold` | Probability threshold for segmentation mask generation. |

## Outputs
| Output File | Description |
|-------------|-------------|
| `probabilities.zarr` | Zarr dataset containing segmentation probabilities (useful for post-processing). |
| `segmentation_mask.zarr` | Segmentation mask in the original image space (before resizing). |
| `data.zarr` | Raw input data used for segmentation (optional). |

## Brain Segmentation Examples

### Sample 771602
https://github.com/user-attachments/assets/1021a3a3-cfa1-460c-b1d7-29746ecf764c

### Sample 782499
https://github.com/user-attachments/assets/66ec0a36-6798-4081-98e4-b6e744355e99

## Installation
To install the package you can use the Dockerfile.

## Contributing

### Code Quality & Testing
Run tests and check code quality using the following tools:

- **Run unit tests with coverage**:
  ```bash
  coverage run -m unittest discover && coverage report
  ```
- **Check documentation coverage**:
  ```bash
  interrogate .
  ```
- **Ensure code follows PEP8 standards**:
  ```bash
  flake8 .
  ```
- **Auto-format code using Black**:
  ```bash
  black .
  ```
- **Sort imports automatically**:
  ```bash
  isort .
  ```

### Pull Requests
- Internal contributors: create a branch.
- External contributors: fork the repository and open a pull request.
- Follow [Angular-style commit messages](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit):
  ```text
  <type>(<scope>): <short summary>
  ```
  **Commit types:**
  - `build`: Build system or dependency changes.
  - `ci`: CI/CD configuration changes.
  - `docs`: Documentation updates.
  - `feat`: New feature.
  - `fix`: Bug fixes.
  - `perf`: Performance improvements.
  - `refactor`: Code restructuring without feature changes.
  - `test`: Adding or updating tests.

## Documentation
To generate API documentation:
```bash
sphinx-apidoc -o doc_template/source/ src
```
Then, build the HTML docs:
```bash
sphinx-build -b html doc_template/source/ doc_template/build/html
```
For more details, refer to the [Sphinx installation guide](https://www.sphinx-doc.org/en/master/usage/installation.html).

---

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)


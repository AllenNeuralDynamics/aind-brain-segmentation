# aind-brain-segmentation

Large-scale segmentation of brain tissue coming from the SmartSPIM microscope. It was trained to segment 2D saggital, coronal or axial planes in channel 639. It performs large-scale segmentation in two ways: lazily if the dataset is too big to load, the images will be resized to 1024x1024, normalized, segmented, upsampled and the results written to a zarr. On the other hand, there is an option to load the dataset in memory if it's not too big (loading a multiscale). This model was trained with a resolution of 14.4 microns in XY and 16 microns in Z.

Input parameters:

- image_path: Path where the OMEZarr is located in S3 or in a local storage. e.g., s3://bucket/SmartSPIM_Dataset/Ex_639_Em_667.zarr
- model_path: Path to where the model is stored.
- output_folder: Folder where results will be written.
- target_size_mb: Target size that will be used in the shared memory compartment for large-scale prediction.
- n_workers: Number of workers that will be pulling data with a pytorch dataloader.
- super_chunksize: Shard that will be pulled from the cloud. This is for optimization in the communication with the bucket.
- scale: The multiscale name that will be used for segmentation, this exists in the OMEZarr if multiple scales were computed.
- scratch_folder: Scratch folder.
- image_height: Image height that will be used to resize. Defualt: 1024
- image_width: Image width that will be used to resize. Defualt: 1024
- prob_threshold: Threshold to cut the probabilities.

The outputs are:

- probabilities.zarr: Zarr with the probabilities, useful for post-processing.
- segmentation_mask.zarr: Segmentation mask of the brain in the original image space (not the resized).
- data.zarr: Raw data used for prediction. This output is optional and not given by default.

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)

## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

To develop the code, run
```bash
pip install -e .[dev]
```

## Contributing

### Linters and testing

There are several libraries used to run linters, check documentation, and run tests.

- Please test your changes using the **coverage** library, which will run the tests and log a coverage report:

```bash
coverage run -m unittest discover && coverage report
```

- Use **interrogate** to check that modules, methods, etc. have been documented thoroughly:

```bash
interrogate .
```

- Use **flake8** to check that code is up to standards (no unused imports, etc.):
```bash
flake8 .
```

- Use **black** to automatically format the code into PEP standards:
```bash
black .
```

- Use **isort** to automatically sort import statements:
```bash
isort .
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repository and open a pull request from the fork. We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:
```text
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect build tools or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bugfix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests

### Documentation
To generate the rst files source files for documentation, run
```bash
sphinx-apidoc -o doc_template/source/ src 
```
Then to create the documentation HTML files, run
```bash
sphinx-build -b html doc_template/source/ doc_template/build/html
```
More info on sphinx installation can be found [here](https://www.sphinx-doc.org/en/master/usage/installation.html).

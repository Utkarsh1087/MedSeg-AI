# Changelog

All notable changes to the **MedSeg-AI** project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-08-15

### Added
- Official v1.0.0 release of the **MedSeg-AI** Medical Image Segmentation Pipeline.
- Modular U-Net architecture implementation (`segmentation_models/unet.py`) optimized for biomedical segmentation tasks.
- Standalone CLI training (`train.py`) and inference (`predict.py`) scripts supporting local and remote environments.
- Interactive Jupyter Notebooks (`train.ipynb`, `Prediction.ipynb`) with automatic environment detection (local filesystem / Google Colab).
- Data augmentation engine featuring random horizontal flips, reflection-padded rotations, and brightness variations.
- Custom Dice Loss and Dice Coefficient evaluation metrics.
- Automatic training visualization exporting loss and Dice coefficient curves.
- Pre-trained model weights (`save_trained_models/unet_trained.h5`).
- Structured dataset and input/output directories with sample medical images and ground truth masks.
- Comprehensive documentation, issue templates, and pull request guidelines.

## [0.2.0] - 2025-07-02

### Added
- Model training pipeline with checkpointing and validation tracking.
- Prediction and post-processing module with binary mask generation and adaptive resizing.

### Changed
- Standardized file paths using dynamic root directory resolution.

## [0.1.0] - 2025-05-20

### Added
- Initial project structure and requirements specification.
- Core U-Net model definition and dataset loading utilities.

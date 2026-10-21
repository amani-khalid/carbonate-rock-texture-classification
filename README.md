# Carbonate Rock Texture Classification Using Wavelet-Based Transfer Learning

Comparing ResNet50, Wavelet Packet Transform (WPT), and Dual-Tree Complex Wavelet Transform (DTCWT) for carbonate rock texture classification from X-ray micro-computed tomography (micro-CT) images.

## Overview

Carbonate rocks contain heterogeneous pore structures and texture patterns at multiple scales. Identifying these textures in micro-CT images can support digital rock characterization and the study of reservoir properties.

This research investigates whether wavelet-based representations improve texture classification compared with a standard image-based convolutional neural network. Three approaches are evaluated using a pretrained ResNet50 backbone:

- **ResNet50 baseline:** classification directly from image inputs.
- **WPT–ResNet50:** classification using energy-selected Haar wavelet packet coefficients.
- **DTCWT–ResNet50:** classification using energy-selected directional wavelet features.

The notebooks contain model implementations, training procedures, cross-validation, evaluation metrics, and visualizations.

## Methodology

The experimental workflow consists of:

1. Initial training using a combined dataset of public rock images and locally labeled micro-CT images.
2. Adaptation to three local texture classes: **Class 1**, **Class 2**, and **Class 3**.
3. Stratified five-fold cross-validation on the local training data.
4. Final model training and evaluation on a separate local test folder.

Local adaptation uses two training stages: training the classification head, followed by unfreezing ResNet50’s final residual block (`layer4`) together with the classifier.

### Model comparison

| Approach | Input representation | Wavelet configuration |
| --- | --- | --- |
| ResNet50 baseline | Image inputs with ImageNet normalization | Not applicable |
| WPT–ResNet50 | Selected wavelet packet coefficients | Three-level Haar WPT; 32 selected channels from 64 subbands |
| DTCWT–ResNet50 | Selected wavelet magnitude and lowpass features | Three-level DTCWT; 8 selected channels from 19 available channels |

The wavelet pipelines resize images to **448 × 448 pixels** and modify ResNet50’s first convolutional layer to accept the selected feature channels.

## Repository Contents

The notebooks are located in the `Models/` directory.

| Notebook | Description |
| --- | --- |
| `baseline_model_with_cv.ipynb` | ResNet50 baseline training, local fine-tuning, cross-validation, and final evaluation |
| `wpt_model.ipynb` | Initial WPT–ResNet50 training with data augmentation and energy-based channel selection |
| `wpt_model_finetune_cv.ipynb` | WPT local fine-tuning, five-fold cross-validation, and final evaluation |
| `dtcwt_model_finetune_cv.ipynb` | DTCWT feature extraction, initial training, local fine-tuning, cross-validation, evaluation, and inference examples |

## Recorded Results

The following results are taken from the saved notebook outputs. They have not been independently reproduced.

### Five-fold cross-validation

| Model | Mean accuracy ± SD | Mean weighted F1 ± SD |
| --- | ---: | ---: |
| ResNet50 baseline | 73.90% ± 6.77% | 0.7301 ± 0.0691 |
| WPT–ResNet50 | 60.65% ± 7.30% | 0.5805 ± 0.1003 |
| DTCWT–ResNet50 | **87.74% ± 6.14%** | **0.8757 ± 0.0633** |

Standard deviations above follow the notebook summaries. The accompanying research report lists different DTCWT standard deviations: 5.97% for accuracy and 0.0616 for weighted F1.

### Final test evaluation

The local test set contains **38 images**: 16 from Class 1, 13 from Class 2, and 9 from Class 3.

| Model | Accuracy | Weighted F1 | Balanced accuracy |
| --- | ---: | ---: | ---: |
| ResNet50 baseline | 78.95% | 0.7932 | 0.8075 |
| WPT–ResNet50 | 55.26% | 0.5080 | 0.5066 |
| DTCWT–ResNet50 | **94.74%** | **0.9476** | **0.9373** |

DTCWT–ResNet50 correctly classified **36 of 38 images**, achieving the highest recorded test accuracy. Its recorded accuracy exceeds the baseline by **15.79 percentage points**.

These comparisons should be interpreted alongside the implementation limitations below.

## Data

The research uses public rock images together with locally prepared carbonate micro-CT patches. The local dataset contains three manually assigned texture classes.

The public dataset referenced in the research report is available at:

[Rock Classification Dataset on Kaggle](https://www.kaggle.com/datasets/stealthtechnologies/rock-classification)

Image loading uses `torchvision.datasets.ImageFolder`, which expects a separate subfolder for each class. For example:

```text
local_data/
├── train/
│   ├── Class 1/
│   ├── Class 2/
│   └── Class 3/
└── test/
    ├── Class 1/
    ├── Class 2/
    └── Class 3/
```

Use matching class-folder names across training and evaluation sets. The original images and trained checkpoints must be supplied separately to reproduce the experiments.

The WPT fine-tuning notebook depends on an initial WPT checkpoint. The baseline and DTCWT notebooks contain both initial training and later adaptation stages.

The code selects CUDA when available and otherwise uses the CPU. Initial loading of ImageNet-pretrained weights requires an internet connection unless the weights are already cached.

## Reproducibility and Limitations

These notebooks retain exploratory code and historical outputs. They are research implementations and may require adjustments for execution from a clean kernel.

- **WPT checkpoint loading:** Saved logs report loading zero tensors from the earlier WPT checkpoint. The initial notebook saves weights under `model_state`, while the fine-tuning loader checks `model_state_dict`. The recorded WPT results therefore do not demonstrate successful transfer of the earlier rock-domain checkpoint. Correcting this requires a new experiment.
- **Baseline test-set selection:** During final baseline training, the test loader is passed to the validation loop for checkpoint selection. Consequently, the recorded baseline test accuracy is not an independent held-out estimate.
- **Dataset independence:** Initial training combines public and local images. Exact split manifests are needed to confirm that final test images, related patches, and augmented variants were excluded from earlier training stages.
- **Small evaluation sample:** The final test set contains 38 images. Larger independent datasets are needed to assess generalization.
- **Notebook execution state:** Some exploratory baseline cells reference variables not defined in the preceding code. Review these cells before using “Run All.”

The recorded results are retained for transparency. Future corrected experiments should be reported separately.

## Research Background

This repository accompanies the senior research project

**Classification of Carbonate Rock Textures from X-ray Micro-tomography Using Wavelet-Based Transfer Learning**



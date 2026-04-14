# UltraSemiNet: Semi-Supervised Fetal Ultrasound Segmentation

This repository contains the official PyTorch implementation of **UltraSemiNet**, a highly efficient and robust framework for semi-supervised medical image segmentation, specifically designed for fetal ultrasound biometry (e.g., Head Circumference).

## Highlights
* **Multi-View Training:** Utilizes Test-Time Augmentation (TTA) inspired views during the self-training process to ensure viewpoint invariance.
* **Spatial-Aware Temperature (SAT) Scaling:** Dynamically scales the calibration temperature based on proximity to the predicted object boundaries, improving confidence calibration on uncertain edges.
* **Pseudo-Label Certainty Masking (PCM):** Filters out highly uncertain predictions in the unlabeled set to prevent the network from confirming its own mistakes.
* **Lightweight & Efficient:** Employs a standard 2D U-Net backbone, achieving real-time inference (>300 FPS) and lower computational training costs compared to complex dual-network approaches like Cross-Teaching or MC-Net.

## Project Structure

```bash
UltraSemiNet/
├── model.py            # U-Net backbone definition
├── components.py       # Core modules (SAT, PCM, CPS Gate, Temp Scaler)
├── losses.py           # Custom Dice and PCM loss objectives
├── dataset.py          # PyTorch Datasets for multi-view semi-supervised loading
├── train.py            # Main training loop with multi-stage optimization
├── eval_hc18.py        # Evaluation script with multi-metric support (px and mm)
├── metrics.py          # Surface Distance calculation utilities (ASD, HD95)
└── transform_utils.py  # Utilities for Test-Time Augmentation (TTA)
```

## Requirements

The code has been tested with the following core dependencies:
* Python 3.8+
* PyTorch 2.0+
* torchvision
* opencv-python (`cv2`)
* numpy
* Pillow

To install simply run:
```bash
pip install torch torchvision opencv-python numpy pillow tqdm
```

## Data Preparation

The code expects the dataset to be in the following structure. By default, it looks for the data in the relative path `./data/`.

```bash
data/
└── FBUI/
    ├── train_images/
    ├── train_masks/
    ├── val_images/
    ...
└── HC18/
    └── test_set/
        ├── 001_HC.png
        ├── 001_HC_Annotation.png
        └── test_set_pixel_size.csv  # Required for evaluating in millimeters
```

## Usage

### 1. Training

To train the UltraSemiNet from scratch using your semi-supervised configuration:

```bash
python train.py --root ./data/FBUI --batch_size 16 --epochs 30 --lr 1e-4 --fold 0
```

### 2. Evaluation

To evaluate the trained model on an external test set like HC18 and compute metrics (DSC, ASD, HD95) in **millimeters**:

```bash
# Evaluate using physical spacing (mm)
python eval_hc18.py --root ./data/HC18/test_set --model_path path/to/ultraseminet_best.pth --unit mm

# Evaluate using pixels
python eval_hc18.py --root ./data/HC18/test_set --model_path path/to/ultraseminet_best.pth --unit px
```

## Acknowledgements

If you find this repository useful, please consider citing our work. For any questions, please open an issue in this repository.

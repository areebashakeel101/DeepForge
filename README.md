# DeepForge 🚀

A comprehensive collection of Deep Learning models implemented in Jupyter Notebooks, covering various computer vision and natural language processing tasks including image classification, object detection, image captioning, and generative models.

## 📋 Table of Contents

- [Overview](#overview)
- [Notebooks](#notebooks)
  - [Image Classification](#image-classification)
  - [Object Detection](#object-detection)
  - [Image Captioning](#image-captioning)
  - [Generative Models](#generative-models)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)

## 🔍 Overview

This repository contains implementations of various deep learning architectures using PyTorch and TensorFlow/Keras frameworks. All notebooks are designed to run on Google Colab with GPU support.

## 📓 Notebooks

### Image Classification

#### 1. AlexNet & VGGNet for Plant Classification
**File:** `ALEXNET_VGGNET_forPlantClassification.ipynb`

- **Description:** Transfer learning implementation using AlexNet and VGGNet architectures for plant species classification
- **Framework:** PyTorch
- **Key Features:**
  - Custom `LimitedImageFolder` dataset class with configurable sample limits per class
  - AlexNet with modified classifier for plant classification
  - VGGNet comparison implementation
  - 70/30 train-test split with data augmentation
  - Early stopping with patience mechanism
  - Confusion matrix visualization and accuracy/loss plots
- **Dataset:** Plant Dataset (Google Drive mounted)

#### 2. ResNet for Plant Classification
**File:** `resnet__forPlantClassification.ipynb`

- **Description:** Fine-tuned ResNet50 model for plant species classification
- **Framework:** TensorFlow/Keras
- **Key Features:**
  - ResNet50 with pretrained ImageNet weights
  - Global Average Pooling with custom dense layers
  - Dropout regularization (0.3)
  - Adam optimizer with learning rate 0.0001
  - Model saving and loading functionality
  - Training visualization with accuracy/loss plots
- **Dataset:** Plant Dataset (50% subset sampling)

---

### Object Detection

#### 3. Faster R-CNN
**File:** `FasterRCNN.ipynb`

- **Description:** Object detection using Faster R-CNN with ResNet50 backbone for pet detection
- **Framework:** PyTorch (torchvision)
- **Key Features:**
  - Custom `PetDataset` class with XML annotation parsing
  - Faster R-CNN with ResNet50-FPN backbone
  - Custom `FastRCNNPredictor` for single-class detection
  - Bounding box visualization
  - Training with custom collate function for variable-size inputs
- **Dataset:** Oxford-IIIT Pet Dataset (annotations in Pascal VOC XML format)

#### 4. YOLO (Fine-tuned)
**File:** `YOLO_Finetuned.ipynb`

- **Description:** Fine-tuned YOLOv8 for pet detection with custom dataset preparation
- **Framework:** Ultralytics YOLOv8
- **Key Features:**
  - XML to YOLO format annotation conversion
  - Custom YAML configuration file generation
  - Train/validation split preparation
  - Dataset visualization with bounding boxes
  - Integration with Ultralytics training pipeline
- **Dataset:** Oxford-IIIT Pet Dataset (converted to YOLO format)

---

### Image Captioning

#### 5. GRU Model for Image Captioning (Fine-tuned)
**File:** `GRU_Model_FineTuned.ipynb`

- **Description:** Image captioning using GRU-based sequence model with MobileNetV2 features
- **Framework:** TensorFlow/Keras
- **Key Features:**
  - Caption preprocessing with start/end tokens
  - Keras Tokenizer for vocabulary building
  - MobileNetV2 feature extraction (1280-dim features)
  - AlexNet feature comparison (9216-dim features)
  - GRU-based decoder with embedding layer
  - Input/target sequence preparation with padding
  - Training loss visualization
- **Dataset:** COCO Captions (mini version)

#### 6. ResNet Model for Image Captioning (Fine-tuned)
**File:** `ResNet_Model_FineTuned_For imagecaptioning.ipynb`

- **Description:** Image captioning using ResNet18 features with GRU decoder
- **Framework:** PyTorch + TensorFlow/Keras hybrid
- **Key Features:**
  - ResNet18 feature extraction (512-dim features)
  - COCO class-based caption generation
  - Label-to-caption conversion from YOLO annotations
  - COCO128 dataset processing
  - Combined PyTorch (features) + Keras (training) pipeline
- **Dataset:** COCO128 dataset

---

### Generative Models

#### 7. GAN (Generative Adversarial Network)
**File:** `GAN.ipynb`

- **Description:** Basic GAN implementation for Fashion-MNIST image generation
- **Framework:** PyTorch
- **Key Features:**
  - Generator with BatchNorm and ReLU activations (4-layer MLP)
  - Discriminator with LeakyReLU activations
  - RMSprop optimizer for both networks
  - Latent dimension: 128
  - Binary Cross Entropy loss
  - Training visualization with generated samples
  - Loss curve plotting (Generator & Discriminator)
- **Dataset:** Fashion-MNIST

---

## 🛠️ Requirements

```
torch>=1.9.0
torchvision>=0.10.0
tensorflow>=2.15.0
keras
ultralytics
numpy
matplotlib
scikit-learn
opencv-python
Pillow
pycocotools
timm
seaborn
```

## 💻 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DeepForge.git
cd DeepForge
```

2. Install dependencies:
```bash
pip install torch torchvision tensorflow keras ultralytics numpy matplotlib scikit-learn opencv-python Pillow pycocotools timm seaborn
```

3. For Google Colab (recommended):
   - Upload notebooks to Google Colab
   - Mount Google Drive for dataset access
   - Enable GPU runtime (Runtime > Change runtime type > GPU)

## 🚀 Usage

### Running on Google Colab (Recommended)

1. Open any notebook in Google Colab
2. Mount your Google Drive when prompted
3. Ensure your dataset is in the correct Google Drive path
4. Run cells sequentially

### Running Locally

1. Ensure you have CUDA-compatible GPU (recommended)
2. Update dataset paths in the notebooks to match your local structure
3. Run using Jupyter Notebook or JupyterLab

## 📁 Datasets

| Dataset | Notebooks | Description |
|---------|-----------|-------------|
| Plant Dataset | AlexNet/VGGNet, ResNet Classification | Plant species images for classification |
| Oxford-IIIT Pet Dataset | Faster R-CNN, YOLO | Pet images with bounding box annotations |
| Fashion-MNIST | GAN | 28x28 grayscale fashion item images |
| COCO Captions | GRU Image Captioning | Image-caption pairs for captioning |
| COCO128 | ResNet Image Captioning | Subset of COCO with YOLO-format labels |

## 📊 Model Architectures Summary

| Model | Task | Input Size | Framework |
|-------|------|------------|-----------|
| AlexNet | Classification | 224x224 | PyTorch |
| VGGNet | Classification | 224x224 | PyTorch |
| ResNet50 | Classification | 224x224 | TensorFlow |
| ResNet18 | Feature Extraction | 224x224 | PyTorch |
| Faster R-CNN | Object Detection | Variable | PyTorch |
| YOLOv8 | Object Detection | Variable | Ultralytics |
| GRU | Sequence Modeling | Variable | TensorFlow |
| GAN | Image Generation | 28x28 | PyTorch |
| MobileNetV2 | Feature Extraction | 224x224 | TensorFlow |

## 📝 License

This project is open-source and available under the MIT License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---

**Note:** All notebooks are designed to run on Google Colab with GPU support. Ensure you have access to the required datasets and sufficient compute resources for training.

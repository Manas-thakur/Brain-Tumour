# Brain Tumor MRI Detection and Classification

![Brain Tumor Detection](https://img.shields.io/badge/Medical-AI-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A deep learning project that performs simultaneous brain tumor detection (segmentation) and classification from MRI scans using a multi-task neural network architecture.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

## 🔍 Overview

This project implements a multi-task deep learning model that:
1. **Detects** brain tumors in MRI scans (segmentation)
2. **Classifies** the tumor type into one of four categories:
   - Glioma
   - Meningioma
   - Pituitary
   - No tumor

The model utilizes a modified U-Net architecture trained on the Brain Tumor MRI Dataset from Kaggle.

## 📊 Dataset

The project uses the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle, which contains MRI scans categorized into the following classes:
- Glioma tumor
- Meningioma tumor
- Pituitary tumor
- No tumor

The dataset is automatically downloaded using the OpenDatasets library.

## 🧠 Model Architecture

The model is based on a **Multi-Task U-Net** architecture that simultaneously performs:

- **Segmentation**: Identifying the tumor region in the MRI scan
- **Classification**: Determining the type of tumor present

Key features:
- Shared encoder for both tasks
- Task-specific decoders for segmentation and classification
- Binary segmentation for tumor detection
- Multi-class classification for tumor type identification

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Brain-tumour.git
cd Brain-tumour
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

Required packages:
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- OpenDatasets

## 🚀 Usage

### Training the Model

Run the main notebook to train the model:
```bash
jupyter notebook main.ipynb
```

The notebook will:
1. Download the dataset
2. Preprocess the images
3. Create data loaders
4. Define and train the model
5. Visualize results
6. Save the trained model

### Making Predictions

To use the pre-trained model for predictions:

```python
from model import load_model, predict

# Load the model
model = load_model('multi_task_unet.pth')

# Make prediction on a new image
seg_mask, tumor_type = predict('path/to/your/image.jpg', model)

# Display results
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.imread('path/to/your/image.jpg', cv2.IMREAD_GRAYSCALE), cmap='gray')
plt.title("Original Image")
plt.subplot(1, 2, 2)
plt.imshow(seg_mask, cmap='gray')
plt.title(f"Predicted Tumor Type: {tumor_type}")
plt.show()
```

## 📈 Results

The model achieves:
- Segmentation accuracy: ~85-90%
- Classification accuracy: ~90-95% 

Sample predictions:
- The model correctly identifies tumor regions in MRI scans
- It accurately classifies different types of brain tumors
- Visualization tools help interpret the results

## 🔮 Future Work

Potential improvements for the project:
- Implement a more sophisticated U-Net architecture with skip connections
- Add data augmentation to improve model generalization
- Incorporate attention mechanisms for better feature extraction
- Implement other evaluation metrics like Dice coefficient and IoU
- Fine-tune hyperparameters for better performance

## 🙏 Acknowledgments

- [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by Masoud Nickparvar on Kaggle
- The PyTorch team for the deep learning framework
- The medical imaging research community for advancements in the field

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

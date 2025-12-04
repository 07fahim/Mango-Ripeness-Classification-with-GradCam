# 🥭 Mango Ripeness Classification with GradCam

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-99%25-success.svg)](README.md)

This project implements deep learning models to classify **six stages of mango ripeness** using transfer learning and hybrid architectures. Additionally, Grad-CAM visualizations are used to interpret the decision-making process of the models.
## 📊 Overview

- **Dataset**: 6,000 mango images (1,000 per class)
  - Training: 4,800 images
  - Validation: 600 images
  - Testing: 600 images
- **Image Size**: 224x224 pixels
- **Batch Size**: 64
- **Models Tested**: 8 different architectures
- **Best Performance**: 99% accuracy (DenseNet201)
- **Tools**: TensorFlow, Keras, scikit-learn, Grad-CAM

## 🎯 Ripeness Stages

The model classifies mangoes into six distinct stages:

1. **Unripe** - Green, firm mangoes
2. **Early Ripe** - Starting to show color changes
3. **Partially Ripe** - Mixed green and yellow/red coloring
4. **Ripe** - Optimal for consumption
5. **Overripe** - Very soft, past prime
6. **Perished** - Spoiled, not suitable for consumption

## ✨ Features

- **Data Augmentation**: Rotation, shifts, zoom, and flips to improve model robustness
- **Smart Callbacks**: 
  - Model checkpointing for saving best weights
  - Early stopping to prevent overfitting
  - Learning rate reduction on plateau
- **Grad-CAM Visualizations**: Interpret what the model "sees" when making predictions
- **Comprehensive Evaluation**: Classification reports and confusion matrices

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/mango-ripeness-classification.git
cd mango-ripeness-classification
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Dataset

The dataset is already included in the repository under the `data/` directory. No additional downloads required!

## 💻 Usage

### Training Models

All model notebooks are located in the `models/` folder. To train a model:

```bash
jupyter notebook models/
```

Available notebooks:
- `mangodensenet201.ipynb` - DenseNet201 
- `mangoinceptionv3.ipynb` - InceptionV3
- `mangomobilenetv2.ipynb` - MobileNetV2
- `mangoxception.ipynb` - Xception
- `mangoxceptionlstm.ipynb` - XceptionLSTM (Best performer)
- `mangocnn.ipynb` - Custom CNN (EfficientNetV2B0)
- `mangovgg19.ipynb` - VGG19
- `mangoresnet50.ipynb` - ResNet50

Each notebook includes:
- Model architecture setup
- Training and validation
- Grad-CAM visualization
- Performance evaluation

## 📈 Model Performance

### Accuracy Comparison

![Model Accuracy Chart](assets/chart.png)

| Rank | Model | Test Accuracy | Training Time | Parameters | Notes |
|------|-------|---------------|---------------|------------|-------|
| 🥇 | **DenseNet201** | **98.02%** | ~45 min | 20M | Best overall performance, dense connections |
| 🥈 | **XceptionLSTM** | **98.33%** | ~50 min | 23M | Sequential modeling capability |
| 🥉 | **InceptionV3** | **98.10%** | ~40 min | 24M | Multi-scale feature extraction |
| 4 | **MobileNetV2** | **98.00%** | ~30 min | 3.5M | Fastest inference, mobile-ready |
| 5 | **Xception** | **98.00%** | ~42 min | 23M | Depthwise separable convolutions |
| 6 | **CNN (EfficientNetV2B0)** | **97.00%** | ~35 min | 7M | Efficient compound scaling |
| 7 | **VGG19** | **97.30%** | ~55 min | 144M | Classic architecture, very deep |
| 8 | **ResNet50** | **92.00%** | ~38 min | 26M | Good baseline with residual connections |


### Confusion Matrix - Xception+Lstm (Best Model)

Results on test set (600 images):

![Confusion Matrix](assets/confusin_matrix.png)

**Key Observations**:
- Most confusion occurs between **Early Ripe** and **Unripe** stages (9 misclassifications)
- **Ripe** stage has perfect classification (100%)
- Overall accuracy: **98.33%**
- Diagonal dominance indicates strong classification performance across all classes

## 🛠️ Requirements

Create a `requirements.txt` file with:

```txt
tensorflow>=2.8.0
keras>=2.8.0
scikit-learn>=1.0.0
numpy>=1.21.0
pandas>=1.3.0
opencv-python>=4.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 📁 Project Structure

```
mango-ripeness-classification/
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
├── models/
│   ├── mangodensenet201.ipynb
│   ├── mangoinceptionv3.ipynb
│   ├── mangomobilenetv2.ipynb
│   ├── mangoxception.ipynb
│   ├── mangoxceptionlstm.ipynb
│   ├── mangocnn.ipynb
│   ├── mangovgg19.ipynb
│   └── mangoresnet50.ipynb
├── saved_models/
├── visualizations/
├── requirements.txt
├── LICENSE
└── README.md
```

## 🎨 Grad-CAM Visualizations

Grad-CAM (Gradient-weighted Class Activation Mapping) helps visualize which parts of the mango image the model focuses on when making predictions. This provides interpretability and helps validate that the model is learning relevant features.

![image](https://github.com/user-attachments/assets/a540502b-54f6-40db-8ca0-9cf2ba303e3e)
![image](https://github.com/user-attachments/assets/c5a1dbcc-66d1-4f1d-99eb-c3eccf563a7b)
![image](https://github.com/user-attachments/assets/a4ac0b3b-9d48-48db-8dee-22d56a9f210c)
![image](https://github.com/user-attachments/assets/ebbb81a2-1f5f-4856-bef9-c83e9d30a471)
![image](https://github.com/user-attachments/assets/5dbe55db-d9ba-4385-b92d-953ff6529fc7)
![image](https://github.com/user-attachments/assets/7ff741d0-3a61-4d9a-b9a3-36cfcf987294)

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a new branch: `git checkout -b feature-name`
3. **Make** your changes and commit: `git commit -m "Add feature"`
4. **Push** to the branch: `git push origin feature-name`
5. **Submit** a pull request

Please ensure your code follows the existing style and includes appropriate tests.

## 📄 License

This project is licensed under the **MIT License** - feel free to modify and distribute.

See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Kaggle Community** for datasets and resources
- **xAI** for inspiration and tools
- **TensorFlow & Keras Teams** for excellent deep learning frameworks
- **Transfer Learning** research community for pre-trained models

## 📧 Contact

For questions or feedback, please open an issue or reach out through GitHub.

## 🔮 Future Work

- [ ] Mobile app deployment
- [ ] Real-time video classification
- [ ] Additional fruit types
- [ ] Model quantization for edge devices
- [ ] Web-based demo interface

---

⭐ **If you find this project useful, please consider giving it a star!** ⭐









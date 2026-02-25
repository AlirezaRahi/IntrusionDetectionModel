## README.md

```markdown
# 🛡️ CNN-BiLSTM Intrusion Detection System for Multi-Class Network Attacks

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-2.13-red.svg)](https://keras.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Elsevier-blueviolet)](https://doi.org/10.1016/j.compeleceng.2024.109892)

<div align="center">
  <img src="data/checkpoints/confusion_matrix.png" alt="Confusion Matrix" width="600"/>
  <p><em>Figure 1: Confusion Matrix of CNN-BiLSTM Model on Farm-Flow Dataset</em></p>
</div>

## 📋 Overview

This repository contains the official implementation of an **Enhanced CNN-Bidirectional LSTM Model for Real-Time Multi-Class Attack Detection in Network Traffic**. The proposed hybrid deep learning framework combines Convolutional Neural Networks (CNN) for spatial feature extraction with Bidirectional Long Short-Term Memory (BiLSTM) networks for temporal dependency modeling, achieving **99.55% overall accuracy** on the Farm-Flow AG-IoT security dataset.

### Key Features:
- 🔍 **Multi-Class Classification**: Detects 8 attack types + normal traffic
- 🕒 **Temporal Modeling**: Bidirectional LSTM captures forward/backward temporal dependencies
- 📊 **Comprehensive Evaluation**: Per-class metrics, confusion matrix, ROC curves
- ⚖️ **Class Imbalance Analysis**: Detailed investigation of minority class performance
- 🚀 **Real-Time Ready**: Optimized for deployment in network security applications

## 📚 Dataset

This project uses the **Farm-Flow | AG-IoT Security Dataset** [1, 2], a comprehensive dataset for intrusion detection in smart agriculture environments.

- **Source**: [Zenodo Repository](https://zenodo.org/records/10964648)
- **Size**: 532 MB, 1,310,000 instances
- **Classes**: 8 attack types + Normal traffic
  - Attack Types: Arp Spoofing, BotNet DDoS, HTTP Flood, ICMP Flood, MQTT Flood, Port Scanning, TCP Flood, UDP Flood
- **Collection Period**: August-October 2022
- **Format**: Network flows (CSV files)

## 🏗️ Architecture

The proposed CNN-BiLSTM architecture consists of:

```
Input (10, 88)
    ↓
├─ Conv1D (64) → BatchNorm → ReLU → MaxPool → Dropout(0.3)
    ↓
├─ Conv1D (128) → BatchNorm → ReLU → MaxPool → Dropout(0.3)
    ↓
├─ Conv1D (256) → BatchNorm → ReLU → MaxPool → Dropout(0.3)
    ↓
├─ BiLSTM (128) → BiLSTM (64) → Dropout(0.4)
    ↓
├─ Dense (256) → Dropout(0.5) → Dense (128) → Dropout(0.4)
    ↓
└─ Dense (9, Softmax)
```

**Total Parameters**: 767,689

## 📊 Results

### Overall Performance
- **Test Accuracy**: 99.55%
- **Macro Avg F1-Score**: 0.55
- **Weighted Avg F1-Score**: 0.99

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Arp_Spoofing | 0.00 | 0.00 | 0.00 | 5 |
| BotNet_DDOS | 0.00 | 0.00 | 0.00 | 13 |
| HTTP_Flood | 0.99 | 1.00 | 1.00 | 16,308 |
| ICMP_Flood | 0.00 | 0.00 | 0.00 | 2 |
| MQTT_Flood | 0.99 | 1.00 | 1.00 | 18,359 |
| Normal | 1.00 | 0.99 | 1.00 | 1,996 |
| Port_Scanning | 0.00 | 0.00 | 0.00 | 44 |
| TCP_Flood | 1.00 | 1.00 | 1.00 | 10,059 |
| UDP_Flood | 1.00 | 1.00 | 1.00 | 11,855 |

<div align="center">
  <img src="data/checkpoints/training_history.png" alt="Training History" width="800"/>
  <p><em>Figure 2: Training and Validation Accuracy/Loss Curves</em></p>
</div>

<div align="center">
  <img src="roc_curves.png" alt="ROC Curves" width="600"/>
  <p><em>Figure 3: ROC Curves for Multi-Class Classification</em></p>
</div>

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- TensorFlow 2.13+
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/AlirezaRahi/IntrusionDetectionModel.git
cd IntrusionDetectionModel
```

2. **Create and activate virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the dataset**
   - Download the Farm-Flow dataset from [Zenodo](https://zenodo.org/records/10964648)
   - Extract to `C:\Users\alire\Downloads\datasets` (or update the path in `main.py`)

### Project Structure

```
IntrusionDetectionModel/
├── src/
│   ├── data_preprocessing.py    # Data loading and preprocessing
│   ├── feature_engineering.py    # Feature extraction and selection
│   ├── model.py                  # CNN-BiLSTM architecture
│   ├── train.py                  # Training pipeline
│   └── evaluate.py               # Evaluation metrics
├── data/
│   ├── processed/                # Saved preprocessors
│   └── checkpoints/              # Model checkpoints and results
├── main.py                        # Main execution script
|
└── README.md                      # This file
```

### Running the Code

1. **Update data paths** in `main.py`:
```python
data_path = r'C:\Users\alire\Downloads\datasets'  # Your dataset path
processed_path = r'C:\Alex\Projects\ai_security_env\Intrusion Detection Model\data\processed'
checkpoint_path = r'C:\Alex\Projects\ai_security_env\Intrusion Detection Model\data\checkpoints'
```

2. **Run the main script**
```bash
python main.py
```

## 📈 Results Visualization

After training, the following visualizations are generated:
- `data/checkpoints/training_history.png` - Accuracy and loss curves
- `data/checkpoints/confusion_matrix.png` - Confusion matrix
- `roc_curves.png` - ROC curves for all classes
- `data/checkpoints/final_results.txt` - Detailed results summary

## 📝 Citation

If you use this code or the model in your research, please cite:

### Paper Citation
```bibtex
@article{rahi2025cnnbilstm,
  title={An Enhanced CNN-Bidirectional LSTM Model for Real-Time Multi-Class Attack Detection in Network Traffic},
  author={Rahi, Alireza},
  journal={},
  year={2026},
  note={Available at: https://github.com/AlirezaRahi/IntrusionDetectionModel}
}
```

### Dataset Citation
```bibtex
@dataset{ferreira2024farmflow,
  author = {Ferreira, Rafael and Bispo, Ivo Afonso and Rabadão, Carlos and Santos, Leonel and Costa, Rogério Luís de C.},
  title = {Farm-Flow | AG-IoT Security: Intrusion Detection in Smart Agriculture Dataset},
  publisher = {Zenodo},
  version = {v1},
  year = {2024},
  doi = {10.5281/zenodo.10964648},
  url = {https://doi.org/10.5281/zenodo.10964648}
}

@article{ferreira2025farmflow,
  title = {Farm-flow dataset: Intrusion detection in smart agriculture based on network flows},
  journal = {Computers and Electrical Engineering},
  volume = {121},
  pages = {109892},
  year = {2025},
  doi = {10.1016/j.compeleceng.2024.109892},
  author = {Rafael Ferreira and Ivo Bispo and Carlos Rabadão and Leonel Santos and Rogério Luís de C. Costa}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The authors of the [Farm-Flow Dataset](https://zenodo.org/records/10964648) for making their valuable resource publicly available
- [TensorFlow](https://tensorflow.org) and [Keras](https://keras.io) teams for the deep learning frameworks

## 📬 Contact

**Alireza Rahi**

[![GitHub](https://img.shields.io/badge/GitHub-AlirezaRahi-blue?style=flat&logo=github)](https://github.com/AlirezaRahi)
[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-Citations-green?style=flat&logo=google-scholar)](https://scholar.google.com/citations?user=I2ASqS0AAAAJ&hl=en)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Alireza%20Rahi-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/alireza-rahi-6938b4154/)

---

<div align="center">
  <b>⭐ Star this repository if you find it useful! ⭐</b>
</div>
```

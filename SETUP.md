# 🧠 Brain Tumor Detection & Classification System

## Project Description

This is an advanced AI-based system designed to detect and classify brain tumors. The project uses two separate models:

1. **Tumor Detection Model**: Analyzes NIfTI scan images to determine whether a brain tumor is present.
2. **Tumor Classification Model**: If a tumor is found, this model predicts the tumor type (Glioma, Meningioma, Pituitary, or Healthy).

## Key Features

- ✅ **Two-stage analysis**: detection first, then classification
- ✅ **High accuracy**: modern CNN architecture
- ✅ **Fast inference**: efficient prediction pipeline
- ✅ **Multi-modal support**: works with both NIfTI and JPG images
- ✅ **Detailed reporting**: confidence scores included

## Datasets

### 1. BraTS2020 Dataset (for detection)
- **Source**: MICCAI BraTS 2020 Challenge
- **Type**: NIfTI (.nii) files
- **Modalities**: FLAIR, T1, T1CE, T2
- **Labels**: tumor masks (.nii segmentation)
- **Location**: `brats20_dataset/BraTS2020_TrainingData/`

### 2. Brain Tumor Classification Dataset (for classification)
- **Source**: Kaggle Brain Tumor Classification Dataset
- **Type**: JPG images
- **Classes**: Glioma Tumor, Meningioma Tumor, No Tumor, Pituitary Tumor
- **Location**: `Classification_dataset/`

## Requirements

- **Python**: 3.8+
- **TensorFlow**: 2.15.0
- **CUDA**: optional, for GPU support
- **RAM**: at least 8GB
- **Disk space**: 10GB+ for datasets

## Installation & Local Environment Setup

### 1. Create a local virtual environment
Create a virtual environment (`.venv`) to isolate project dependencies:
```bash
# Navigate to the project root directory
cd "Brain Tumer"

# Create the virtual environment
python3 -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows (Command Prompt):
# .venv\Scripts\activate
# On Windows (PowerShell):
# .venv\Scripts\Activate.ps1
```

### 2. Install all requirements
Install all the required libraries using the `requirements.txt` file:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Clone the project
```bash
git clone <repository-url>
cd brain-tumor-detector
```

### 3. Download the datasets
- Place the BraTS dataset inside `brats20_dataset/`
- Place the classification dataset inside `Classification_dataset/`

## Model Training

### Model 1: Tumor Detection
```bash
python train_model1.py
```
This script trains a binary classification model using the BraTS dataset.

### Model 2: Tumor Classification
```bash
python train_model2.py
```
This script trains a multi-class classification model using the Kaggle dataset.

> Note: GPU is recommended for faster training.

## Usage

### 1. Demo in Jupyter Notebook
```bash
jupyter notebook final_output.ipynb
```

### 2. Manual usage
```python
from final_output import load_detector, analyze_nifti_scan

detector = load_detector()
analyze_nifti_scan('path/to/nifti/file.nii', slice_index=75)
```

### 3. From command line
```bash
python -c "
import sys
sys.path.append('.')
from final_output import analyze_nifti_scan
analyze_nifti_scan('brats20_dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Validation_001_flair.nii')
"
```

## Model Architecture

### Detection Model
- **Architecture**: CNN with Global Average Pooling
- **Input**: 128x128x3 RGB image
- **Output**: binary output (tumor or no tumor)
- **Layers**: Conv2D, MaxPooling, Dense, Dropout

### Classification Model
- **Architecture**: MobileNetV2 with custom head
- **Input**: 224x224x3 RGB image
- **Output**: 4 classes (Glioma, Meningioma, Healthy, Pituitary)
- **Transfer learning**: ImageNet weights

## Results & Performance

### Detection Model
- **Accuracy**: ~95%
- **Precision**: ~94%
- **Recall**: ~96%

### Classification Model
- **Accuracy**: ~92%
- **Precision**: ~91%
- **Recall**: ~90%

## File Structure
```
brain-tumor-detector/
├── README.md                           # this file
├── train_model1.py                     # detection model training
├── train_model2.py                     # classification model training
├── final_output.ipynb                  # main demo notebook
├── brain_tumor_detector_final.keras    # trained detection model
├── tumor_classifier_final.keras        # trained classification model
├── brats20_dataset/                    # BraTS dataset
└── Classification_dataset/             # classification dataset
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

## Troubleshooting

### Common issues
1. **"Model weights not found"**: make sure the `.keras` files are in the project root
2. **"NIfTI file not found"**: verify dataset paths are correct
3. **"CUDA out of memory"**: reduce batch size or use CPU mode
4. **"Import errors"**: install all required dependencies

### GPU support
```bash
pip install tensorflow-gpu==2.15.0
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to your branch (`git push origin feature/new-feature`)
5. Open a pull request

## License

This project is licensed under the MIT License.

## Contact

If you have questions or need help, open an issue or contact via email.

---

**Note**: This AI tool is not medical diagnosis software. It is for educational purposes only. Consult a doctor for real medical diagnosis.

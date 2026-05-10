# Rice Variety Classification - End-to-End Pipeline

## Project Information

- **Course Name:** [COURSE_NAME]
- **Course Code:** [COURSE_CODE]
- **Semester:** [SEMESTER]
- **Instructor:** [INSTRUCTOR]

### Team Members
| Student ID | Full Name |
|---|---|
| 2353081 | Võ Nhật Tân |
| 2353318 | Huỳnh Nguyễn Quốc Việt |
| 2352703 | Trần Đào Phúc Long |
| 2352752 | Nguyễn Mạnh Trí Minh |
| 2353059 | Lương Bảo Tài |

## Overview
This repository contains a modular Python pipeline and an end-to-end Google Colab notebook for the classification of 20 Bangladeshi rice varieties. 

The pipeline performs:
1. **Exploratory Data Analysis (EDA):** Visualising class distributions, extracting manual features (color, sharpness, morphology).
2. **Deep Feature Extraction:** Using pretrained backbones (VGG16, ResNet18, ViT-B/16).
3. **ML Benchmarking:** Training and evaluating Logistic Regression, SVM, KNN, Random Forest, and XGBoost on the extracted features.
4. **Deep Learning:** Training custom PyTorch Softmax classifier heads on the extracted features.

Dataset: [An Image Dataset of Rice Varieties (Mendeley)](https://data.mendeley.com/datasets/3mn9843tz2/3)

## Repository Structure

```
riceclassifier/
├── notebooks/
│   └── rice_classifier.ipynb       # Main Colab notebook (Run All compatible)
├── modules/
│   ├── __init__.py
│   ├── config.py                   # Global configuration
│   ├── data_loader.py              # PyTorch Dataset & DataLoader utilities
│   ├── models.py                   # Pretrained backbones
│   ├── data_extractor.py           # Feature extraction script
│   ├── ml_pipeline.py              # Classical ML benchmarks
│   ├── deep_learning.py            # Deep learning classification heads
│   └── utils.py                    # Shared utilities
├── reports/                        # Student-created PDF reports go here
├── features/                       # Output directory for extracted .npy features
├── models/                         # Output directory for trained model checkpoints
├── results/                        # Output directory for benchmark results
├── data/                           # Extracted dataset
├── README.md                       
└── requirements.txt                
```

## How to Run in Google Colab
1. Upload the `notebooks/rice_classifier.ipynb` notebook to Google Colab.
2. Ensure the `modules/` folder is uploaded or cloned into the Colab environment.
3. Select **Runtime -> Run all**.
4. The notebook will automatically download the dataset from Mendeley, perform EDA, run feature extraction, train all ML/DL models, and output the final comparisons.

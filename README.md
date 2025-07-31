# League of Legends match outcome prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-orange?logo=pytorch)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2%2B-green?logo=scikit-learn)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

_Created by **Réka Gábosi**_

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Project Structure](#-project-structure)
- [Model Performance](#model-performance)
- [Visualizations](#visualizations)
- [Acknowledgements](#acknowledgements)

## Project Description

This project applies logistic regression using PyTorch to predict match outcomes in League of Legends based on in-game statistics. It was completed as the **Final project** for the IBM course _Introduction to Neural Networks and Python_.

The project includes:
- Data preprocessing and normalization
- Logistic regression model definition and training
- L2 regularization
- Model evaluation (accuracy, classification report, confusion matrix, ROC curve, etc.)
- Hyperparameter tuning
- Feature importance analysis

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Afhrodite/League-of-Legends-match-outcome-prediction.git
   cd League-of-Legends-match-outcome-prediction
   ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the project:**
    You can either:
    - Open and run the Jupyter notebook final_project.ipynb for screenshots and tasks, OR
    - Run the polished Python script:

            ```bash
            python model1.py
            ```

## Project Structure

```
├── data/                          # Contains the League of Legends dataset CSV file  
│   └── league_data.csv  
├── images/                        # Contains all images used in the README  
│   ├── Confusion Matrix.png  
│   ├── Feature Importance(Logistic Regression).png  
│   └── Receiver Operating Characteristic.png  
├── Jupyter_notebook/             # Original IBM final project notebook  
│   └── LoL_final_project.ipynb  
├── model.py                      # Polished standalone Python script with all project logic  
├── requirements.txt              # Python dependencies for running the project  
├── LICENSE                       # Project license (Apache 2.0)  
└── README.md                     # This file  
```

## Model Performance

| Version              | Test Accuracy |
|----------------------|---------------|
| Base Logistic Model  | 50.5%         |
| With L2 Regularizer  | 49.5%         |
| After Tuning         | **57.0%**     |

- **Train Accuracy**: ~54.2%  
- **Best Learning Rate**: `0.05`  
- ✅ Model saved and reloaded successfully using `torch.save()` and `torch.load()`

## Visualizations

| Visualization | Preview |
|---------------|---------|
| **Confusion Matrix** | ![Confusion Matrix](images/Confusion%20Matrix.png) |
| **Feature Importance (Logistic Regression)** | ![Feature Importance](images/Feature%20Importance%28Logistic%20Regression%29.png) |
| **Receiver Operating Characteristic** | ![ROC Curve](images/Receiver%20Operating%20Characteristic.png) |

## Acknowledgements

Special thanks to **IBM** for offering the course *Introduction to Neural Networks and Python*, which served as the foundation for this project.

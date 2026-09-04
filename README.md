# Machine Failure Prediction 🤖

A machine learning classification project that analyzes operational and environmental machine parameters to predict whether a machine is likely to fail.

## 📌 Project Overview

Machine failure prediction can help identify potential failures before they occur, supporting preventive maintenance and reducing unexpected downtime.

This project explores machine-related parameters and builds multiple machine learning classification models to predict the `fail` target variable.

## 🎯 Objective

The main objective is to:

* Analyze machine sensor and operational data
* Perform data preprocessing and exploratory data analysis
* Study relationships between machine parameters and failure
* Engineer additional features
* Train and compare multiple machine learning classification models
* Identify the best-performing model for machine failure prediction

## 📊 Dataset

The dataset contains **944 records** and **10 columns**.

### Features

| Feature       | Description                                |
| ------------- | ------------------------------------------ |
| `footfall`    | Footfall-related machine parameter         |
| `tempMode`    | Temperature mode                           |
| `AQ`          | Air quality                                |
| `USS`         | USS sensor measurement                     |
| `CS`          | CS sensor measurement                      |
| `VOC`         | VOC sensor measurement                     |
| `RP`          | RP-related parameter                       |
| `IP`          | IP-related parameter                       |
| `Temperature` | Temperature measurement                    |
| `fail`        | Target variable indicating machine failure |

The target variable is:

```text
fail
```

## 🔍 Project Workflow

```text
Data Loading
     ↓
Data Inspection
     ↓
Data Preprocessing
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Train/Test Split
     ↓
Model Training
     ↓
Model Comparison
     ↓
Best Model Selection
     ↓
Confusion Matrix & Classification Report
```

## 📈 Exploratory Data Analysis

The notebook includes several visual analyses, including:

* Footfall distribution by temperature mode
* Failure distribution
* Pairwise feature relationships
* 3D visualization of machine parameters
* Failure probability analysis across features
* Threshold-based analysis
* Feature engineering using `footfall × tempMode`

A new feature called `nf` was created:

```python
data['nf'] = data['footfall'] * data['tempMode']
```

## 🤖 Machine Learning Models

The project compares multiple classification algorithms, including:

* LightGBM
* Multi-Layer Perceptron
* Bagging Classifier
* Gradient Boosting Classifier
* AdaBoost Classifier
* XGBoost
* Logistic Regression
* Random Forest
* K-Nearest Neighbours
* Support Vector Machine
* Stacking Classifier

## 📊 Model Performance

Based on the recorded notebook results:

| Model                        |   Accuracy |
| ---------------------------- | ---------: |
| Gradient Boosting Classifier | **88.38%** |
| Random Forest Classifier     | **88.38%** |
| Logistic Regression          |     88.03% |
| AdaBoost Classifier          |     87.32% |
| Stacking Classifier          |     86.97% |
| Bagging Classifier           |     84.86% |
| MLP Classifier               |     83.45% |
| K-Nearest Neighbours         |     72.18% |

## 🏆 Best Model

The best-performing model recorded in the notebook is:

**Gradient Boosting Classifier**

### Accuracy

```text
88.38%
```

### Classification Report

| Class       | Precision |   Recall | F1-Score |
| ----------- | --------: | -------: | -------: |
| 0           |      0.91 |     0.88 |     0.90 |
| 1           |      0.85 |     0.89 |     0.87 |
| **Overall** |  **0.88** | **0.88** | **0.88** |

The model was trained using **660 samples** and evaluated on **284 test samples**.

## 🧩 Confusion Matrix

The notebook also generates a confusion matrix for the selected best model to visualize correct and incorrect predictions for machine failure and non-failure cases.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Plotly
* Scikit-learn
* LightGBM
* XGBoost
* Jupyter Notebook

## 🚀 How to Run

Clone the repository:

```bash
git clone https://github.com/Adityarana5855/Machine-Failure-Prediction.git
```

Navigate to the project directory:

```bash
cd Machine-Failure-Prediction
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn lightgbm xgboost jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Machine_Failure_Prediction.ipynb
```

## 📁 Project Structure

```text
Machine-Failure-Prediction/
│
├── Machine_Failure_Prediction.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## 🔗 Repository

GitHub:

https://github.com/Adityarana5855/Machine-Failure-Prediction

## 👨‍💻 Author

**Aditya Pratap Singh**

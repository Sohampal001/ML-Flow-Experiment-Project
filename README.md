============================================================
PROJECT TITLE
============================================================
ML Experiment Tracking and Reproducible Pipeline using
Cookiecutter, Hydra, DVC, and MLflow

============================================================
PROJECT OVERVIEW
============================================================

This project demonstrates a complete Machine Learning workflow
for training, tracking, and comparing models in a reproducible
way. The system uses a Cookiecutter template to standardize the
project structure and integrates modern ML tools to manage
configuration, datasets, and experiments.

Key technologies used in this project:

• Cookiecutter – Generates a reusable machine learning project
  template with a predefined folder structure.

• Hydra – Provides structured configuration management through
  YAML files and allows easy parameter overrides.

• MLflow – Tracks experiments including model parameters,
  evaluation metrics, trained models, and artifacts.

• DVC (Data Version Control) – Tracks datasets and builds
  reproducible data pipelines.

The project trains and compares multiple machine learning models
including Logistic Regression, Random Forest, SVM, and XGBoost
on the Wine Quality dataset.

============================================================
WORKFLOW OF THE PROJECT
============================================================

1. Project Template Creation
   A Cookiecutter template is used to automatically generate a
   standardized ML project structure including configuration,
   source code, and data folders.

2. Dataset Versioning
   The Wine Quality dataset is stored in the data/raw folder and
   tracked using DVC to ensure reproducibility.

3. Data Preprocessing
   The preprocessing script performs the following steps:
   • Load raw dataset
   • Convert target column into binary classification
   • Split dataset into training and testing sets
   • Apply feature scaling using StandardScaler
   • Save processed datasets

4. DVC Pipeline Execution
   The preprocessing stage is defined in dvc.yaml so the pipeline
   can be reproduced automatically when data or scripts change.

5. Model Training
   The training script loads the processed dataset and trains
   different models such as:
   • Logistic Regression
   • XGBoost
   • Random Forest
   • Support Vector Machine (SVM)

6. Experiment Tracking
   During training, MLflow logs:
   • Hyperparameters
   • Evaluation metrics
   • Trained models
   • Feature importance plots
   • Confusion matrix

7. Model Comparison
   MLflow allows comparing multiple runs and determining which
   model performs best.

============================================================
DATASET VERSIONING WITH DVC
============================================================

DVC is used to track datasets and build reproducible pipelines.
Each pipeline stage follows a simple flow:

INPUTS  →  DEPENDENCIES  →  OUTPUTS

Example preprocessing stage:

Command
    python xgboost_project/src/preprocess.py

Inputs
    xgboost_project/data/raw/winequality-red.csv
    (raw dataset)

Dependencies
    xgboost_project/src/preprocess.py
    (data preprocessing script)

Outputs
    xgboost_project/data/preprocessed/winequality-red-preprocessed_train.csv
    xgboost_project/data/preprocessed/winequality-red-preprocessed_test.csv

Process Explanation

1. The raw dataset is tracked by DVC.
2. The preprocess script loads the dataset and prepares it for
   machine learning.
3. The script generates preprocessed train and test datasets.
4. DVC monitors changes in inputs or dependencies.
5. If anything changes, DVC automatically re-runs the pipeline.

Useful Commands

Initialize DVC
    dvc init

Track dataset
    dvc add data/raw/winequality-red.csv

Reproduce pipeline
    dvc repro

============================================================
MODEL TRAINING AND EXPERIMENT TRACKING
============================================================

The training pipeline uses Hydra configuration files to define
the model type and hyperparameters.

During execution the script:

1. Loads configuration using Hydra.
2. Loads the processed training dataset.
3. Trains the selected model.
4. Logs experiment information to MLflow.

MLflow records:

• Parameters (hyperparameters)
• Metrics (accuracy, precision, recall, log_loss)
• Feature importance
• Confusion matrix
• Trained model artifacts

MLflow UI can then be used to visualize and compare experiments.

============================================================
KEY BENEFITS OF THIS PROJECT
============================================================

• Reproducible ML pipeline
• Automated experiment tracking
• Dataset version control
• Configurable training via Hydra
• Easy comparison of machine learning models

============================================================
AUTHOR
============================================================

Soham Pal
Machine Learning Experiment Tracking Project
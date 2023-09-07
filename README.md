# Loan Classification with Machine Learning

## Overview

This repository contains a machine learning analysis focused on loan classification. The analysis aims to predict loan health and identify high-risk loans using machine learning models. This README provides an overview of the project and how to use it.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Analysis](#analysis)
  - [Data](#data)
  - [Machine Learning Models](#machine-learning-models)
- [Results](#results)

## Getting Started

### Prerequisites

- Python 3.x
- Required Python packages (see [requirements.txt](requirements.txt))

1. Clone this repository:
   ```sh
   git clone https://github.com/your-username/loan-classification.git

   cd loan-classification

   import numpy as np
   import pandas as pd
   from pathlib import Path
   from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression

pip install -r requirements.txt

Analysis
Data
The analysis is based on financial data related to loans. The primary goal is to predict whether a loan is healthy (0) or high-risk (1).

Machine Learning Models
Two machine learning models were used for this analysis:

Machine Learning Model 1: Developed using the first dataset.
Machine Learning Model 2: Developed using the second dataset.
Both models were trained using logistic regression and addressed class imbalance using random oversampling.

Results
The analysis produced the following results for both machine learning models:

Machine Learning Model 1:
Balanced Accuracy: 0.95
Precision for Healthy Loans (Class 0): 1.00
Recall for Healthy Loans (Class 0): 0.99
F1-Score for Healthy Loans (Class 0): 1.00
Precision for High-Risk Loans (Class 1): 0.85
Recall for High-Risk Loans (Class 1): 0.91
F1-Score for High-Risk Loans (Class 1): 0.88

Machine Learning Model 2:
Balanced Accuracy: 0.94
Precision for Healthy Loans (Class 0): 1.00
Recall for Healthy Loans (Class 0): 0.99
F1-Score for Healthy Loans (Class 0): 1.00
Precision for High-Risk Loans (Class 1): 0.85
Recall for High-Risk Loans (Class 1): 0.91
F1-Score for High-Risk Loans (Class 1): 0.88



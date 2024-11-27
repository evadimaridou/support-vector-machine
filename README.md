# Neural Networks Assignment 2: Support Vector Machines

This repository contains implementations and experiments with **Support Vector Machines (SVMs)** for classification tasks. The focus is on understanding SVMs by implementing them from scratch and comparing the results with Scikit-learn's library.

## Project Overview

Support Vector Machines are supervised learning models used for classification and regression analysis. This project explores:
- Linear SVM implemented from scratch.
- Kernel SVM with different kernels (Linear, Polynomial, RBF).
- Comparison with Scikit-learn's SVM implementation.

The dataset used for testing is a subset of **CIFAR-10**, focusing on binary classification between two classes.

## Repository Contents

- `linear_svm_from_scratch.py`: Implements a linear SVM using **Gradient Descent**.
- `kernel_svm.py`: Implements an SVM with kernel functions (Linear, Polynomial, RBF).
- `sklearn_svm.py`: Demonstrates the use of Scikit-learn's SVM.
- `report.pdf`: Detailed report on the project, including theoretical background, implementation details, and results.

## Key Features

1. **Custom Implementations**:
   - Linear SVM: Built from scratch to understand the mathematical and algorithmic foundations.
   - Kernel SVM: Includes support for Polynomial and Radial Basis Function kernels.

2. **Comparison with Libraries**:
   - Validates the custom implementations against Scikit-learn's SVM.

3. **Dataset**:
   - The CIFAR-10 dataset, reduced to a binary classification problem.

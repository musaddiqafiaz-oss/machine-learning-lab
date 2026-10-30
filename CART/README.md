
# CART Algorithm Implementation

## Introduction
CART stands for Classification and Regression Trees. It is a supervised machine learning algorithm used for classification and regression tasks.

## Project Objective
The objective of this project is to predict whether a student may pass or fail based on study hours and attendance percentage.

## Technologies Used
- Python
- Scikit-learn
- Matplotlib

## Algorithm Used
This project uses the CART classification algorithm with the Gini impurity criterion to build a decision tree.

## Features
- Takes study hours as input.
- Takes attendance percentage as input.
- Predicts whether a student may pass or fail.
- Displays a decision tree.
- Saves the decision tree as an image.

## Installation
Install the required libraries using:

```bash
python -m pip install scikit-learn matplotlib
```

## How to Run
1. Download or clone this repository.
2. Open the project folder in VS Code.
3. Run the following command:

```bash
python cart_model.py
```

4. Enter the study hours.
5. Enter the attendance percentage.
6. View the prediction and decision tree.

## Example
Input:
- Study Hours: 6
- Attendance: 85%

Output:
- Prediction: Student may PASS.

## Project Files
- `cart_model.py` — Python implementation of the CART algorithm.
- `requirements.txt` — Required Python libraries.
- `cart_tree.png` — Generated decision tree image.

## Disclaimer
This project uses sample data for educational purposes. Predictions are not a reliable measure of actual student performance.

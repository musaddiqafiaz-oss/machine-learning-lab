# ID3 Decision Tree Algorithm

A Python implementation of the **ID3 (Iterative Dichotomiser 3) algorithm** for classification using Entropy and Information Gain. This project uses the classic Play Tennis dataset to demonstrate how a decision tree learns from data and makes predictions.

## Project Overview

The ID3 algorithm is a decision tree learning algorithm used in Machine Learning for classification tasks. It selects the best feature at each step using Information Gain and recursively builds a decision tree.

This project implements the core algorithm from scratch using Python.

## Objectives

- Understand the working of the ID3 decision tree algorithm.
- Calculate Entropy to measure uncertainty in a dataset.
- Calculate Information Gain to select the best splitting feature.
- Build a decision tree recursively.
- Predict class labels for new samples.
- Evaluate the model using training accuracy.

## Technologies Used

- **Python** — algorithm implementation
- **Pandas** — dataset handling
- **Math** — logarithmic calculations for Entropy
- **Collections (Counter)** — class frequency calculations
- **CSV** — dataset storage

## Project Structure

```text
ML lab/
├── Id3 implementation.py
├── play_tennis.csv
├── README.md
└── requirements.txt
```

## Dataset

The project uses the classic Play Tennis dataset, which contains weather conditions and whether tennis should be played.

**Features:**
- Outlook
- Temperature
- Humidity
- Wind

**Target variable:** `PlayTennis` (Yes or No)

The dataset contains 14 examples: 9 positive examples (Yes) and 5 negative examples (No).

## Algorithm

### 1. Entropy

Entropy measures the uncertainty or impurity in a dataset.

\[
H(S)=-\sum_{i=1}^{n}p_i\log_2(p_i)
\]

Where \(p_i\) is the proportion of examples belonging to class \(i\).

### 2. Information Gain

Information Gain measures how much uncertainty is reduced by splitting the dataset using a feature.

\[
IG(S,A)=H(S)-\sum_{v\in Values(A)}
\frac{|S_v|}{|S|}H(S_v)
\]

The feature with the highest Information Gain is selected for the split.

### 3. Decision Tree Construction

The algorithm selects the best feature, divides the dataset into subsets, and repeats the process until a stopping condition is reached.

### 4. Prediction

The learned decision tree is used to predict whether tennis should be played for a given set of weather conditions.

## Installation and Execution

### Prerequisites

Install Python 3 and the required library, Pandas.

```bash
python -m pip install -r requirements.txt
```

### Run the Program

Open a terminal in the project folder and run:

```bash
python "Id3 implementation.py"
```

The program calculates Entropy and Information Gain, builds the decision tree, makes a prediction, and reports training accuracy. It also generates or updates `play_tennis.csv` from the dataset defined in the Python script.

## Expected Results

For the classic Play Tennis dataset, the expected Information Gain values are approximately:

| Feature | Information Gain |
|---|---:|
| Outlook | 0.2467 |
| Humidity | 0.1518 |
| Wind | 0.0481 |
| Temperature | 0.0292 |

**Root node:** Outlook

The program also displays a decision tree, a sample prediction, and training accuracy. The exact displayed output depends on the implementation.

## Limitations

- The project uses a small, classic dataset for educational purposes.
- Training accuracy does not guarantee performance on unseen data.
- The implementation uses categorical features and does not include pruning.
- Unseen feature values may not have a corresponding branch in the learned tree.

## Conclusion

This project demonstrates the fundamental working of the ID3 decision tree algorithm. It shows how Entropy and Information Gain help select features, construct a decision tree, and make classification predictions.

The project provides a practical foundation for understanding decision tree learning in Machine Learning.

---

**Project Type:** Machine Learning Lab  
**Algorithm:** ID3 Decision Tree  
**Language:** Python  
**Dataset:** Play Tennis
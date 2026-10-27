# C4.5 Decision Tree Algorithm

## Overview

This project demonstrates an educational implementation of the C4.5 decision tree learning algorithm using Python. It uses Entropy, Information Gain, and Gain Ratio to select features and build a classification tree.

C4.5 is an extension of ID3 that introduces Gain Ratio for feature selection and supports additional capabilities in its full form, including continuous attributes and missing-value handling.

## Objectives

- Understand the C4.5 decision tree algorithm.
- Calculate Entropy and Information Gain.
- Calculate Split Information and Gain Ratio.
- Construct a decision tree recursively.
- Predict class labels for new examples.
- Evaluate training accuracy.

## Technologies

- Python
- Pandas
- Math
- Collections (`Counter`)

## Dataset

The project uses the classic Play Tennis dataset.

**Input features:** Outlook, Temperature, Humidity, and Wind.

**Target variable:** `PlayTennis` (Yes or No).

## Gain Ratio

Gain Ratio is calculated as:

\[
\text{Gain Ratio}(A)=
\frac{\text{Information Gain}(A)}
{\text{Split Information}(A)}
\]

It helps select a feature while accounting for how the data is divided by that feature.

## Project Structure

```text
02-C45/
├── c45_implementation.py
├── play_tennis.csv
└── README.md
```

## Installation

Install the required library:

```bash
python -m pip install pandas
```

## Run the Program

```bash
python c45_implementation.py
```

The program displays entropy, information gain, split information, gain ratio, the generated tree, a sample prediction, and training accuracy.

## Implementation Scope

This is an educational, categorical-data implementation of C4.5 concepts. It does not implement all features of the full C4.5 algorithm, such as continuous-value splitting, missing-value handling, or pruning.

## Conclusion

This project provides practical experience with decision-tree classification and demonstrates how Gain Ratio can be used to select features during tree construction.

**Language:** Python  
**Algorithm:** C4.5 Decision Tree  
**Dataset:** Play Tennis
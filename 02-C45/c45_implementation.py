
import math
import pandas as pd
from collections import Counter


# -----------------------------------------
# 1. Load the Play Tennis Dataset
# -----------------------------------------

data = pd.read_csv("play_tennis.csv")

TARGET = "PlayTennis"
FEATURES = [col for col in data.columns if col != TARGET]


# -----------------------------------------
# 2. Calculate Entropy
# -----------------------------------------

def entropy(labels):
    counts = Counter(labels)
    total = len(labels)

    result = 0

    for count in counts.values():
        probability = count / total
        result -= probability * math.log2(probability)

    return result


# -----------------------------------------
# 3. Calculate Information Gain
# -----------------------------------------

def information_gain(data, feature, target):
    parent_entropy = entropy(data[target])
    total = len(data)
    weighted_entropy = 0

    for value, subset in data.groupby(feature):
        weight = len(subset) / total
        weighted_entropy += weight * entropy(subset[target])

    return parent_entropy - weighted_entropy


# -----------------------------------------
# 4. Calculate Split Information
# -----------------------------------------

def split_information(data, feature):
    total = len(data)
    result = 0

    for _, subset in data.groupby(feature):
        probability = len(subset) / total
        result -= probability * math.log2(probability)

    return result


# -----------------------------------------
# 5. Calculate Gain Ratio
# -----------------------------------------

def gain_ratio(data, feature, target):
    gain = information_gain(data, feature, target)
    split_info = split_information(data, feature)

    if split_info == 0:
        return 0

    return gain / split_info


# -----------------------------------------
# 6. Build the Decision Tree
# -----------------------------------------

def c45(data, features, target):

    # If all examples have the same class
    if data[target].nunique() == 1:
        return data[target].iloc[0]

    # If no features remain, return the majority class
    if not features:
        return data[target].mode().iloc[0]

    # Calculate Gain Ratio for every available feature
    scores = {
        feature: gain_ratio(data, feature, target)
        for feature in features
    }

    best_feature = max(scores, key=scores.get)

    # If no useful split exists, return the majority class
    if scores[best_feature] <= 0:
        return data[target].mode().iloc[0]

    tree = {best_feature: {}}
    remaining_features = [
        feature for feature in features
        if feature != best_feature
    ]

    # Recursively create branches
    for value, subset in data.groupby(best_feature):
        tree[best_feature][value] = c45(
            subset,
            remaining_features,
            target
        )

    return tree


# -----------------------------------------
# 7. Predict Using the Decision Tree
# -----------------------------------------

def predict(tree, sample):

    if not isinstance(tree, dict):
        return tree

    feature = next(iter(tree))
    value = sample.get(feature)

    branches = tree[feature]

    if value not in branches:
        return None

    return predict(branches[value], sample)


# -----------------------------------------
# 8. Run the Algorithm
# -----------------------------------------

print("\n========== C4.5 DECISION TREE ==========")

print("\nDataset:")
print(data)

print("\nEntropy of the dataset:")
print(round(entropy(data[TARGET]), 4))

print("\nInformation Gain:")
for feature in FEATURES:
    print(
        f"{feature}: "
        f"{information_gain(data, feature, TARGET):.4f}"
    )

print("\nSplit Information:")
for feature in FEATURES:
    print(
        f"{feature}: "
        f"{split_information(data, feature):.4f}"
    )

print("\nGain Ratio:")
for feature in FEATURES:
    print(
        f"{feature}: "
        f"{gain_ratio(data, feature, TARGET):.4f}"
    )

decision_tree = c45(data, FEATURES, TARGET)

print("\nGenerated Decision Tree:")
print(decision_tree)

sample = {
    "Outlook": "Sunny",
    "Temperature": "Cool",
    "Humidity": "Normal",
    "Wind": "Strong"
}

prediction = predict(decision_tree, sample)

print("\nSample:")
print(sample)

print("\nPrediction:", prediction)

# Evaluate training accuracy
correct = 0

for _, row in data.iterrows():
    example = row[FEATURES].to_dict()
    actual = row[TARGET]
    predicted = predict(decision_tree, example)

    if predicted == actual:
        correct += 1

accuracy = (correct / len(data)) * 100

print(f"\nTraining Accuracy: {accuracy:.2f}%")

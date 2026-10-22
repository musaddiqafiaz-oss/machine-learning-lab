
import math
import pandas as pd
from collections import Counter

# ---------------------------------
# 1. Create the Play Tennis dataset
# ---------------------------------

data = pd.DataFrame({
    "Outlook": [
        "Sunny", "Sunny", "Overcast", "Rain",
        "Rain", "Rain", "Overcast", "Sunny",
        "Sunny", "Rain", "Sunny", "Overcast",
        "Overcast", "Rain"
    ],
    "Temperature": [
        "Hot", "Hot", "Hot", "Mild",
        "Cool", "Cool", "Cool", "Mild",
        "Cool", "Mild", "Mild", "Mild",
        "Hot", "Mild"
    ],
    "Humidity": [
        "High", "High", "High", "High",
        "Normal", "Normal", "Normal", "High",
        "Normal", "Normal", "Normal", "High",
        "Normal", "High"
    ],
    "Wind": [
        "Weak", "Strong", "Weak", "Weak",
        "Weak", "Strong", "Strong", "Weak",
        "Weak", "Weak", "Strong", "Strong",
        "Weak", "Strong"
    ],
    "PlayTennis": [
        "No", "No", "Yes", "Yes",
        "Yes", "No", "Yes", "No",
        "Yes", "Yes", "Yes", "Yes",
        "Yes", "No"
    ]
})

# Save the dataset as a CSV file
data.to_csv("play_tennis.csv", index=False)

print("PLAY TENNIS DATASET")
print(data.to_string(index=False))
print("\nTotal records:", len(data))


# ---------------------------------
# 2. Calculate entropy
# ---------------------------------

def entropy(labels):
    counts = Counter(labels)
    total = len(labels)
    result = 0.0

    for count in counts.values():
        probability = count / total
        result -= probability * math.log2(probability)

    return result


# ---------------------------------
# 3. Calculate information gain
# ---------------------------------

def information_gain(data, feature, target):
    parent_entropy = entropy(data[target])
    weighted_entropy = 0.0
    total = len(data)

    for value in data[feature].unique():
        subset = data[data[feature] == value]

        weighted_entropy += (
            len(subset) / total
        ) * entropy(subset[target])

    return parent_entropy - weighted_entropy


# ---------------------------------
# 4. Build the ID3 decision tree
# ---------------------------------

def id3(data, features, target):
    labels = data[target]

    # Stop if all examples have the same class
    if labels.nunique() == 1:
        return labels.iloc[0]

    # If no features remain, use majority class
    if not features:
        return labels.mode().iloc[0]

    # Select feature with maximum information gain
    gains = {
        feature: information_gain(data, feature, target)
        for feature in features
    }

    best_feature = max(gains, key=gains.get)

    tree = {best_feature: {}}

    remaining_features = [
        feature for feature in features
        if feature != best_feature
    ]

    # Recursively create branches
    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]

        tree[best_feature][value] = id3(
            subset,
            remaining_features,
            target
        )

    return tree


# ---------------------------------
# 5. Predict a new example
# ---------------------------------

def predict(tree, sample):
    while isinstance(tree, dict):
        feature = next(iter(tree))
        value = sample[feature]

        # Handle an unseen feature value
        if value not in tree[feature]:
            return "Unknown"

        tree = tree[feature][value]

    return tree


# ---------------------------------
# 6. Train and display the model
# ---------------------------------

target = "PlayTennis"
features = [
    "Outlook", "Temperature", "Humidity", "Wind"
]

print("\nENTROPY OF THE DATASET:")
print(round(entropy(data[target]), 4))

print("\nINFORMATION GAIN:")
for feature in features:
    gain = information_gain(data, feature, target)
    print(f"{feature}: {gain:.4f}")

tree = id3(data, features, target)

print("\nLEARNED ID3 DECISION TREE:")
print(tree)


# ---------------------------------
# 7. Test with new weather conditions
# ---------------------------------

sample = {
    "Outlook": "Sunny",
    "Temperature": "Cool",
    "Humidity": "Normal",
    "Wind": "Strong"
}

prediction = predict(tree, sample)

print("\nNEW WEATHER CONDITIONS:")
print(sample)
print("Prediction:", prediction)


# ---------------------------------
# 8. Evaluate training accuracy
# ---------------------------------

correct = 0

for _, row in data.iterrows():
    sample = row[features].to_dict()
    actual = row[target]
    predicted = predict(tree, sample)

    if actual == predicted:
        correct += 1

accuracy = correct / len(data) * 100

print("\nTRAINING EVALUATION:")
print(f"Correct predictions: {correct}/{len(data)}")
print(f"Training accuracy: {accuracy:.2f}%")

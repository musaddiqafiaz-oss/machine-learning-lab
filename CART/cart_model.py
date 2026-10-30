
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Training data: [study_hours, attendance_percentage]
X = [
    [1, 40],
    [2, 50],
    [2, 60],
    [3, 65],
    [4, 70],
    [5, 80],
    [6, 85],
    [7, 90],
    [8, 95],
    [3, 45]
]

# Results: 0 = Fail, 1 = Pass
y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0]

# Create CART model
model = DecisionTreeClassifier(
    criterion="gini",
    random_state=42,
    max_depth=3
)

# Train the model
model.fit(X, y)

print("CART Model Trained Successfully!")

# Take input from the user
hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance percentage: "))

# Make prediction
prediction = model.predict([[hours, attendance]])

if prediction[0] == 1:
    print("Prediction: Student may PASS.")
else:
    print("Prediction: Student may FAIL.")

# Display the decision tree
plt.figure(figsize=(12, 7))

plot_tree(
    model,
    feature_names=["Study Hours", "Attendance"],
    class_names=["Fail", "Pass"],
    filled=True
)

plt.title("CART Decision Tree")
plt.savefig("cart_tree.png")
plt.show()

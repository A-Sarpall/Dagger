# from sklearn import tree
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# import pandas as pd

# # Load the CSV file
# file_path = "C:/Users/sruja/Downloads/transaction_data_corrected.csv"  # Replace with your file path
# df = pd.read_csv(file_path)

# # Separate the features (first 3 columns) and labels (last column)
# X = df.iloc[:, :3].values.tolist()  # Convert first 3 columns to list of lists
# Y = df.iloc[:, -1].values.tolist()  # Convert last column to a list

# # Example input for prediction
# Xs = [
#     [350, 2, 75],   # Example 1
#     [250, 23, 50],  # Example 2
#     [500, 3, 100],  # Example 3
#     [200, 22, 45],  # Example 4
#     [450, 1, 90]    # Example 5
# ]


# # Logistic Regression
# log_reg = LogisticRegression(max_iter=1000)  # Ensure convergence with max_iter
# log_reg.fit(X, Y)
# log_reg_prediction = log_reg.predict(Xs)
# print(f"Logistic Regression Prediction: {log_reg_prediction}")

from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = "data.csv"  # Replace with your file path
df = pd.read_csv(file_path)

# Separate the features (first 3 columns) and labels (last column)
X = df.iloc[:, :3].values.tolist()  # Convert first 3 columns to list of lists
Y = df.iloc[:, -1].values.tolist()  # Convert last column to a list

# Example input for prediction
Xs = [
    [350, 2, 75],   # Example 1
    [250, 23, 50],  # Example 2
    [500, 3, 100],  # Example 3
    [200, 22, 45],  # Example 4
    [450, 1, 90]    # Example 5
]

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)  # Ensure convergence with max_iter
log_reg.fit(X, Y)
log_reg_prediction = log_reg.predict(Xs)
print(f"Logistic Regression Prediction: {log_reg_prediction}")

# Convert predictions to DataFrame for visualization
test_cases = pd.DataFrame(Xs, columns=["Transaction_Amount", "Time_of_Day", "Distance_From_Home"])
test_cases["Prediction"] = log_reg_prediction

# Scatter Plot: Transaction Amount vs. Distance from Home
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df, 
    x="Transaction_Amount", 
    y="Distance_From_Home", 
    hue="Label", 
    style="Label", 
    palette="Set1", 
    alpha=0.7
)
plt.scatter(
    test_cases["Transaction_Amount"], 
    test_cases["Distance_From_Home"], 
    color="black", 
    label="Test Cases", 
    marker="X", 
    s=100
)
plt.title("Transaction Amount vs. Distance from Home")
plt.xlabel("Transaction Amount")
plt.ylabel("Distance from Home")
plt.legend()
plt.show()

# Box Plot: Transaction Amount by Logistic Regression Predictions
plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df, 
    x="Label", 
    y="Transaction_Amount", 
    palette="Set2"
)
plt.title("Transaction Amount by Logistic Regression Predictions")
plt.xlabel("Label")
plt.ylabel("Transaction Amount")
plt.show()

# # Heatmap: Correlation Matrix of Dataset
# correlation_matrix = df[["Transaction_Amount", "Time_of_Day", "Distance_From_Home"]].corr()
# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
# plt.title("Correlation Matrix of Features")
# plt.show()

# Heatmap: Mean Values of Indicators by Fraud/Legit
heatmap_data = df.groupby("Label")[["Transaction_Amount", "Time_of_Day", "Distance_From_Home"]].mean()
plt.figure(figsize=(8, 4))
sns.heatmap(
    heatmap_data,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Mean Values of Indicators by Fraud/Legit")
plt.xlabel("Indicators")
plt.ylabel("Label")
plt.show()


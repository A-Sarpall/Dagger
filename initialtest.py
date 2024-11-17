from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd

#[height, weight, shoe size]
#X = [
    #[500, 14, 20],  # $500 transaction at 2 PM, 20 miles from home
    #[3000, 3, 500], # $3000 transaction at 3 AM, 500 miles from home
    #[200, 9, 5],    # $200 transaction at 9 AM, 5 miles from home
    #[150, 16, 3],   # $150 transaction at 4 PM, 3 miles from home
    #[10000, 23, 1000], # $10,000 transaction at 11 PM, 1000 miles from home
    #[50, 12, 1],    # $50 transaction at 12 PM, 1 mile from home
    #[800, 20, 15],  # $800 transaction at 8 PM, 15 miles from home
    #[1500, 2, 300], # $1500 transaction at 2 AM, 300 miles from home
    #[75, 10, 2],    # $75 transaction at 10 AM, 2 miles from home
    #[1200, 21, 10], # $1200 transaction at 9 PM, 10 miles from home
    #[2500, 6, 400]  # $2500 transaction at 6 AM, 400 miles from home
#]

#Y = [
    #'legit', 'fraud', 'legit', 'legit', 
    #'fraud', 'legit', 'legit', 'fraud', 
    #'legit', 'legit', 'fraud'
#]

# Load the CSV file
file_path = "large_fraud_detection_data.csv"  # Replace with your file path
df = pd.read_csv(file_path)

# Separate the features (first 7 columns) and labels (last column)
X = df.iloc[:, :7].values.tolist()  # Convert first 7 columns to list of lists
Y = df.iloc[:, -1].values.tolist()  # Convert last column to a list

Xs = [[6342, 8, 75, 0, 1, 3, 900]]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

prediction = clf.predict(Xs)

print (prediction)


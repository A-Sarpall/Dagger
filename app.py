import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for matplotlib

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

# Route to serve the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to serve the graphs page
@app.route('/graphs')
def graph():
    # Generate the graphs before rendering the page
    generate_graphs()
    return render_template('graphs.html')

# Route to handle CSV uploads (simulated)
@app.route('/upload', methods=['POST'])
def upload_csv():
    # Simulating an upload without actually processing a file
    return jsonify({"message": "File uploaded successfully!"})

# Route to serve specific graph images
@app.route('/static/graph_<int:id>.png')
def get_graph(id):
    file_path = f'static/graph_{id}.png'
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png')
    else:
        return "Graph not found", 404

# Function to generate graphs
def generate_graphs():
    # Path to your dataset
    file_path = "data.csv"  # Replace with your actual file path
    if not os.path.exists(file_path):
        print("CSV file not found. Ensure 'data.csv' is in the project directory.")
        return

    df = pd.read_csv(file_path)

    # Separate features and labels
    X = df.iloc[:, :3].values.tolist()
    Y = df.iloc[:, -1].values.tolist()

    # Example input for predictions
    Xs = [
        [350, 2, 75],
        [250, 23, 50],
        [500, 3, 100],
        [200, 22, 45],
        [450, 1, 90]
    ]

    # Logistic Regression
    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X, Y)
    log_reg_prediction = log_reg.predict(Xs)

    # Generate Scatter Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Transaction_Amount", y="Distance_From_Home", hue="Label", style="Label", palette="Set1", alpha=0.7)
    plt.scatter(
        [x[0] for x in Xs],
        [x[2] for x in Xs],
        color="black",
        label="Test Cases",
        marker="X",
        s=100
    )
    plt.title("Transaction Amount vs. Distance from Home")
    plt.xlabel("Transaction Amount")
    plt.ylabel("Distance from Home")
    plt.legend()
    plt.savefig("static/graph_1.png")
    plt.close()

    # Generate Box Plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="Label", y="Transaction_Amount", palette="Set2")
    plt.title("Transaction Amount by Logistic Regression Predictions")
    plt.xlabel("Label")
    plt.ylabel("Transaction Amount")
    plt.savefig("static/graph_2.png")
    plt.close()

    # Generate Heatmap
    heatmap_data = df.groupby("Label")[["Transaction_Amount", "Time_of_Day", "Distance_From_Home"]].mean()
    plt.figure(figsize=(8, 4))
    sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Mean Values of Indicators by Fraud/Legit")
    plt.xlabel("Indicators")
    plt.ylabel("Label")
    plt.savefig("static/graph_3.png")
    plt.close()

if __name__ == '__main__':
    # Ensure the 'static' directory exists for saving graphs
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)

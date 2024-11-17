#app.py
from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Read CSV and generate dummy fraud insights
            df = pd.read_csv(file_path)
            threshold = 10000
            fraud_alerts = df[df['amount'] > threshold].to_dict(orient='records')
            stats = {
                "total_transactions": len(df),
                "potential_fraud": len(fraud_alerts),
                "average_amount": df['amount'].mean()
            }
            
            return jsonify({"alerts": fraud_alerts, "stats": stats})
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

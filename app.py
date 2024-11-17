from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/graphs', methods=['GET', 'POST'])
def graphs():
    if request.method == 'POST':
        # Get the data from the form submission
        client_name = request.form.get('input1', '')
        withdrawal_amount_str = request.form.get('input2', '')
        time = request.form.get('input3', '')
        distance_from_home_str = request.form.get('input4', '')

        # Check if withdrawal_amount or distance_from_home are empty
        if not withdrawal_amount_str or not distance_from_home_str:
            return "Error: Withdrawal Amount and Distance From Home are required fields."

        try:
            # Convert the input values to integers
            withdrawal_amount = int(withdrawal_amount_str)
            distance_from_home = int(distance_from_home_str)
        except ValueError:
            return "Error: Please enter valid integers for Withdrawal Amount and Distance From Home."

        # Store the data in a dictionary
        data = {
            'client_name': client_name,
            'withdrawal_amount': withdrawal_amount,
            'time': time,
            'distance_from_home': distance_from_home,
        }

        # Render the graph page and pass the data to it
        return render_template('graphs.html', data=data)

    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)

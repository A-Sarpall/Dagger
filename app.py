from flask import Flask, render_template

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/graphs', methods=['GET', 'POST'])
def graphs():
    # Pass the image filenames to the template
    images = [
        "graph_1.png",  # Path relative to the static folder
        "graph_2.png",
        "graph_3.png"
    ]
    
    # Render the graph page and pass the image paths to it
    return render_template('graphs.html', images=images)

if __name__ == '__main__':
    app.run(debug=True)

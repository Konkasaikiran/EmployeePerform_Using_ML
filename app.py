from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("Training/gwp.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = [float(request.form[i]) for i in request.form]
    final_input = np.array([data])
    result = model.predict(final_input)[0]
    return render_template('submit.html', prediction=round(result, 2))

if __name__ == "__main__":
    app.run(debug=True)

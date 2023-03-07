from flask import Flask, render_template,request
from app.utils import Prediction 

app = Flask(__name__)

@app.route('/')
def start():
    return render_template("car_html.html")

@app.route('/predict', methods = ["POST","GET"])
def predict_price():
    data = request.form
    pred_obj = Prediction()
    predicted_price = pred_obj.predict_price(data)
    print(predicted_price)
    
    return str(predicted_price)


if __name__ == "__main__":
    app.run(debug=True, port = 5050, host='127.0.0.100')
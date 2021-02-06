#importing the necessary dependencies
from flask import Flask, render_template, request,jsonify
from flask_cors import cross_origin,CORS
import flask_cors
import pickle
import numpy as np
import sklearn
app = Flask(__name__) # initializing a flask app
model=pickle.load(open("car_price_prediction1.pkl","rb"))

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")
@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        init_values=[float(x) for x in request.form.values()]
        final_values=list(np.array(init_values))
        result=model.predict(np.array(final_values).reshape(1,9))
        return render_template("results.html", prediction=result[0])
if __name__ == "__main__":
    app.run(debug=True)
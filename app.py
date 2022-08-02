import numpy as np
from flask import Flask, request, jsonify, render_template


import pickle


app = Flask(__name__)
model = pickle.load(open('knn_titanic_model.pkl', 'rb'))



@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    P = float(request.args.get('P'))
    Sex = request.args.get('Sex')
    Age = float(request.args.get('Age'))
    Sibsp = float(request.args.get('Sibsp'))
    Par = float(request.args.get('Par'))
    F =   float(request.args.get('F'))
    
    prediction = model.predict([[P,Sex,Age,Sibsp,Par,F]])
    
        
    if prediction==0:
      return render_template('index.html',prediction_text='ups! person not survived')
    else:
      return render_template('index.html',prediction_text='ooh yee! person  survived')
    
if __name__=="__main__":
  app.run(debug=True)

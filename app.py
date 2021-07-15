import flask
from flask import Flask, render_template
import pickle
import pandas as pd


#######  XGB MODEL #######
# Use pickle to load in the pre-trained model
with open(f'model/xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Set up the main route

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/test")
def test():
    return render_template("test.html")


@app.route('/main', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        age = flask.request.form['age']
        gender = flask.request.form['gender']
        height = flask.request.form['height']
        weight = flask.request.form['weight']
        bp_hi = flask.request.form['bp_hi']
        bp_lo = flask.request.form['bp_lo']
        cholesterol = flask.request.form['cholesterol']
        gluc = flask.request.form['gluc']
        smoke = flask.request.form['smoke']
        alco = flask.request.form['alco']
        active = flask.request.form['active']
        # Make DataFrame for model
        input_variables = pd.DataFrame([[age, gender, height,weight, bp_hi,bp_lo, cholesterol,gluc,smoke,alco,active]],
                                       columns=['age', 'gender', 'height','weight','bp_hi','bp_lo','cholesterol','gluc','smoke','alco','active'],
                                       dtype=float,
                                       index=['input'])

        # Get the model's prediction
        
        prediction = model.predict(input_variables)[0]
        if prediction == 0:
            first = "You are not at Risk!"

        else:
            first = "You are at Risk!"
    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('main.html',
                                     original_input={'Age':age,
                                                     'Gender':gender,
                                                     'Height':height,
                                                     'Weight':weight,
                                                     'Systolic BP':bp_hi,
                                                     'Diastolic BP':bp_lo,
                                                     'Cholesterol':cholesterol,
                                                     'gluc':gluc,
                                                     'smoke':smoke,
                                                     'alco':alco,
                                                     'active':active},
                                     result=first
                                     )

#######  RF MODEL #######

with open(f'model/rf_model.pkl', 'rb') as m:
    rfmodel = pickle.load(m)

@app.route('/further', methods=['GET', 'POST'])
def further():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('further.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        age = flask.request.form['age']
        sex = flask.request.form['sex']
        cp = flask.request.form['cp']
        trestbps = flask.request.form['trestbps']
        chol = flask.request.form['chol']
        fbs = flask.request.form['fbs']
        restecg = flask.request.form['restecg']
        thalach = flask.request.form['thalach']
        exang = flask.request.form['exang']
        oldpeak = flask.request.form['oldpeak']
        slope = flask.request.form['slope']
        ca = flask.request.form['ca']
        thal = flask.request.form['thal']
        # Make DataFrame for model
        input_variables = pd.DataFrame([[age, sex, cp, trestbps, chol,fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]],
                                       columns=['age', 'sex', 'cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'],
                                       dtype=float,
                                       index=['input'])

        # Get the model's prediction
        
        prediction = rfmodel.predict(input_variables)[0]
        if prediction == 0:
            second = "No Heart Disease, But at Risk!!"

        else:
            second = "You may have Heart Disease!"
    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('further.html',
                                     original_input={'Age':age,
                                                     'Sex':sex,
                                                     'cp':cp,
                                                     'trestbps':trestbps,
                                                     'chol':chol,
                                                     'fbs':fbs,
                                                     'restecg':restecg,
                                                     'thalach':thalach,
                                                     'exang':exang,
                                                     'oldpeak':oldpeak,
                                                     'slope':slope,
                                                     'ca':ca,
                                                     'thal':thal},
                                     result=second
                                     )

if __name__ == '__main__':
    app.run()

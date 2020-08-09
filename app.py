from flask import Flask, render_template, request, redirect, g, session, url_for
from flask_uploads import UploadSet, configure_uploads, IMAGES
from functools import wraps
import flask_login
import csv
import cv2
import os
import numpy as np
import pandas as pd
import shutil
import pickle
import joblib
import diseasePrediction
import liverModel
from tensorflow.keras.models import load_model
from keras.models import load_model as keras_load_model
from keras.preprocessing.image import  load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

class User:
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

    def __repr__(self):
        return f'<User: {self.username}>'
users = []
users.append(User(id=1, username='Aditya', password='8619131789'))
users.append(User(id=2, username='Darsh', password='9998218437'))
users.append(User(id=3, username='Kaushal', password='8460711971'))


app = Flask(__name__)
photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = '.'
configure_uploads(app, photos)
app.secret_key = 'aditya8619131789'
def liverModelSecondPredict(features):
    model = pickle.load(open('model/liverDisease2.pkl', 'rb'))
    stats = pickle.load(open('model/liver_Disease2.pkl', 'rb'))
    stats['mean'].insert(1, 0)
    stats['std'].insert(1, 1)
    mean = np.array(stats['mean'])
    std = np.array(stats['std'])
    final_features = (np.array(features) - mean) / std
    prediction = model.predict(np.array(final_features))
    output = int(round(prediction[0][0]))
    if (output > 0):
        prediction = 'You may have Liver disease with 75% probability'
    else:
        prediction = 'You may not have Liver disease'
    return prediction

def lungModelPredict(file):
    pathImage = './image'
    if os.path.isdir(pathImage):
        shutil.rmtree(pathImage)
        os.makedirs(pathImage)
    else:
        os.makedirs(pathImage)
    if file:
        filename = photos.save(file)
        os.rename(filename, pathImage + '/' + filename)
    image = cv2.imread(pathImage + '/' + filename)
    img = cv2.resize(image, (512, 512))
    img = img_to_array(img)
    model = load_model('model/lungCancer.h5')
    model.load_weights('model/lungCNNCancer.h5')
    resp = model.predict_classes(img.reshape((1, 512, 512, 3)), batch_size=8, verbose=0)
    if resp[0] == [1]:
        result = 'Not Cancer'
    elif resp[0] == [0]:
        result = 'Cancer'
    return result

login_manager = flask_login.LoginManager()
login_manager.init_app(app)
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if g.user is None:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

@login_required
@app.before_request
def before_request():
    g.user = None

    if 'user_id' in session:
        user = [x for x in users if x.id == session['user_id']][0]
        g.user = user
        
@app.route('/')
def beforeLogin():
    session.pop('user_id', None)
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = str(request.form['name'])
        password = str(request.form['mobile'])
        print(username)
        print(password)
        print()
        user = [x for x in users if x.username == username][0]
        print(user.username)
        print(user.password)
        if user.username == username and user.password == password:
            session['user_id'] = user.id
            return redirect(url_for('home'))

        return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/index')
def home():
    if not g.user:
        return redirect(url_for('login'))

    return render_template('index.html')

@app.route("/logout")
@login_required
def logout():
    session.pop('user_id', None)
    return redirect(url_for('beforeLogin'))

@app.route('/check')
@login_required
def check():
    return render_template("checkups.html")

@app.route('/lungDisease')
@login_required
def lungDisease():
    return render_template("lungDisease.html")

@app.route('/lungDiseasePrediction1')
@login_required
def lungDiseasePrediction1():
    return render_template("lungDiseasePrediction1.html")


############ Image oriented ###################
@app.route('/lungDiseaseResult1', methods=['POST'])
@login_required
def lungDiseaseResult1():
    if request.method == 'POST':
        file = request.files['photo']
        result = lungModelPredict(file)
        return render_template('lungDiseaseResult1.html', result=result)
    return render_template("lungDiseasePrediction1.html")

@app.route('/lungDiseasePrediction2')
@login_required
def lungDiseasePrediction2():
    return render_template("lungDiseasePrediction2.html")

@app.route('/malariaDisease')
@login_required
def malariaDisease():
    return render_template("malariaDisease.html")

@app.route('/diabetesDisease')
@login_required
def diabetesDisease():
    return render_template("diabetesDisease.html")

@app.route('/breastCancerDisease')
@login_required
def breastCancerDisease():
    return render_template("breastCancerDisease.html")


@app.route('/liverDisease')
@login_required
def liverDisease():
    return render_template("liverDisease.html")

@app.route('/liverDiseasePrediction1')
@login_required
def liverDiseasePredictionBefore():
    return render_template("liverDiseasePredictionBefore.html")

@app.route('/liverDiseasePrediction1', methods=['POST'])
@login_required
def liverDiseasePrediction1():
    if request.method == 'POST':
        return render_template("liverDiseasePrediction.html")

@app.route('/liverDiseaseResult1', methods=['POST'])
@login_required
def liverDiseaseResult1():
    if request.method == 'POST':
        test1 = int(request.form['mcv'])
        test2 = int(request.form['alkphos'])
        test3 = int(request.form['sgpt'])
        test4 = int(request.form['sgot'])
        test5 = int(request.form['gammagt'])
        test6 = int(request.form['drinks'])
        test = test1, test2, test3, test4, test5, test6
        result = liverModel.liverModel_rfc(test)
        return render_template('liverDiseaseResult.html', temp=result)
    return render_template("liverDiseasePredictionBefore.html")

@app.route('/liverDiseasePrediction2')
@login_required
def liverDiseasePredictionBefore2():
    return render_template("liverDiseasePredictionBefore2.html")

@app.route('/liverDiseasePrediction2', methods=['POST'])
@login_required
def liverDiseasePrediction2():
    if request.method == 'POST':
        return render_template("liverDiseasePrediction2.html")

@app.route('/liverDiseaseResult2', methods=['POST'])
@login_required
def liverDiseaseResult2():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        result = liverModelSecondPredict(features)
        return render_template('liverDiseaseResult2.html', prediction_text=result)
    return render_template("liverDiseasePredictionBefore2.html")

@app.route('/heartbeat')
@login_required
def heartbeat():
    return render_template("heartbeatReport.html")

@app.route('/heartDisease')
@login_required
def heartDisease():
    return render_template("heartDisease.html")

@app.route('/heartDiseasePrediction1')
@login_required
def heartDiseasePrediction1():
    return render_template("heartDiseasePrediction.html")

@app.route('/heartDiseaseResult1', methods=['POST'])
@login_required
def heartDiseaseResult1():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = int(request.form['exang'])
        cp = int(request.form['cp'])
        fbs = float(request.form['fbs'])
        x = np.array([age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang]).reshape(1, -1)
        scaler_path = os.path.join(os.path.dirname(__file__), 'model/heartDieases.pkl')
        scaler = None
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        x = scaler.transform(x)
        model_path = os.path.join(os.path.dirname(__file__), 'model/heartDieases.sav')
        clf = joblib.load(model_path)
        y = clf.predict(x)
        if y == 0:
            return render_template('heartDiseaseResultNo.html')
        else:
            return render_template("heartDiseaseResult.html", stage=int(y))
    return render_template("heartDiseasePrediction.html")


@app.route('/heartDiseasePrediction2')
@login_required
def heartDiseasePrediction2():
    return render_template("heartDiseasePrediction2.html")

@app.route('/heartDiseaseResult2', methods=['POST'])
@login_required
def heartDiseaseResult2():
    if request.method == 'POST':
        model = open("model/heartDieases2.pkl", "rb")
        clfr = joblib.load(model)
        parameters = []
        parameters.append(request.form['age'])
        parameters.append(request.form['sex'])
        parameters.append(request.form['cp'])
        parameters.append(request.form['trestbps'])
        parameters.append(request.form['chol'])
        parameters.append(request.form['fbs'])
        parameters.append(request.form['restecg'])
        parameters.append(request.form['thalach'])
        parameters.append(request.form['exang'])
        parameters.append(request.form['oldpeak'])
        parameters.append(request.form['slope'])
        parameters.append(request.form['ca'])
        parameters.append(request.form['thal'])
        inputFeature = np.asarray(parameters).reshape(1, -1)
        my_prediction = clfr.predict(inputFeature)
        return render_template('heartDiseaseResult2.html', prediction=int(my_prediction[0]))
    return render_template("heartDiseasePrediction2.html")







@app.route('/Symptom-to-disease-prediction')
@login_required
def symptomToDiseasePrediction():
    return render_template("Symptom-to-disease-prediction.html",symptoms=symptoms)

@app.route('/symptom_disease_predict', methods=['POST'])
@login_required
def symptom_disease_predict():
    selected_symptoms = []
    if(request.form['Symptom1']!="") and (request.form['Symptom1'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom1'])
    if(request.form['Symptom2']!="") and (request.form['Symptom2'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom2'])
    if(request.form['Symptom3']!="") and (request.form['Symptom3'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom3'])
    if(request.form['Symptom4']!="") and (request.form['Symptom4'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom4'])
    if(request.form['Symptom5']!="") and (request.form['Symptom5'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom5'])

    disease = diseasePrediction.dosomething(selected_symptoms)
    return render_template('symptomd_isease_predict.html',disease=disease,symptoms=selected_symptoms)







@app.route('/covidcheck')
@login_required
def covidcheck():
    return render_template("covidCheckups.html")

@app.route('/covidXray')
@login_required
def covidXray():
    return render_template("covidXrayCheck.html")

############ Image oriented ###################
@app.route('/covidXrayResult',methods = ['POST', 'GET'])
@login_required
def covidXrayResult():
    if request.method == 'POST':
        file = request.files['photo']
        pathImage = './image'
        if os.path.isdir(pathImage):
            shutil.rmtree(pathImage)
            os.makedirs(pathImage)
        else:
            os.makedirs(pathImage)
        if file:
            # filename = secure_filename(file.filename)
            filename = photos.save(request.files['photo'])
            os.rename(filename, pathImage+'/'+filename)
        inception_chest = load_model('model/inceptionv3_chest.h5')
        image = cv2.imread(pathImage+'/'+filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = np.array(image) / 255
        image = np.expand_dims(image, axis=0)

        inception_pred = inception_chest.predict(image)
        probability = inception_pred[0]
        # print("Inception Predictions:")
        if probability[0] > 0.7:
            inception_chest_pred = str('%.2f' % (probability[0] * 100) + '% COVID')
        else:
            inception_chest_pred = str('%.2f' % ((1 - probability[0]) * 100) + '% NonCOVID')
        print(inception_chest_pred)
        if os.path.isdir(pathImage):
            shutil.rmtree(pathImage)
            os.makedirs(pathImage)
        else:
            os.makedirs(pathImage)
        return render_template('covidXrayResult.html',chest_pred=inception_chest_pred)
    return render_template("covidXrayCheck.html")


@app.route('/covidCt')
@login_required
def covidCt():
    return render_template("covidCtCheck.html")

############ Image oriented ###################
@app.route('/covidCtResult',methods = ['POST', 'GET'])
@login_required
def covidCtResult():
    if request.method == 'POST':
        file = request.files['photo']
        pathImage = './image'
        if os.path.isdir(pathImage):
            shutil.rmtree(pathImage)
            os.makedirs(pathImage)
        else:
            os.makedirs(pathImage)
        if file:
            # filename = secure_filename(file.filename)
            filename = photos.save(request.files['photo'])
            os.rename(filename, pathImage+'/'+filename)
        xception_ct = load_model('model/xception_ct.h5')
        image = cv2.imread(pathImage+'/'+filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = np.array(image) / 255
        image = np.expand_dims(image, axis=0)

        xception_pred = xception_ct.predict(image)
        probability = xception_pred[0]
        print("Xception Predictions:")
        if probability[0] > 0.7:
            xception_ct_pred = str('%.2f' % (probability[0] * 100) + '% COVID')
        else:
            xception_ct_pred = str('%.2f' % ((1 - probability[0]) * 100) + '% NonCOVID')
        print(xception_ct_pred)
        if os.path.isdir(pathImage):
            shutil.rmtree(pathImage)
            os.makedirs(pathImage)
        else:
            os.makedirs(pathImage)
        return render_template('covidCtResult.html',ct_pred=xception_ct_pred)
    return render_template("covidCtCheck.html")



@app.route('/covidSymptom')
@login_required
def covidSymptom():
    return render_template("covidSymptomCheck.html")

@app.route('/covidSymptomResult',methods = ['POST', 'GET'])
@login_required
def covidSymptomResult():
    if request.method == 'POST':
        parameters = []
        parameters.append(int(request.form['Fever']))
        parameters.append(int(request.form['Tiredness']))
        parameters.append(int(request.form['Dry-Cough']))
        parameters.append(int(request.form['Difficulty-in-Breathing']))
        parameters.append(int(request.form['Sore-Throat']))
        parameters.append(int(request.form['Pains']))
        parameters.append(int(request.form['Nasal-Congestion']))
        parameters.append(int(request.form['Runny-Nose']))
        parameters.append(int(request.form['Diarrhea']))
        age = str(request.form['Age'])
        for i in range(len(age)):
            parameters.append(int(age[i]))
        gender = str(request.form['Gender'])
        for i in range(len(gender)):
            parameters.append(int(gender[i]))
        parameters.append(int(request.form['Contact']))
        para = []
        para.append(parameters)
        filename = 'model/covid_model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        res = loaded_model.predict(para)
        if res[0]==1:
            result = "COVID"
        else:
            result = "NO COVID"
        return render_template("covidSymptomResult.html",result=result)
    return render_template("covidSymptomCheck.html")


@app.route('/covidTracker')
@login_required
def covidtra():
    return render_template("covidIndex.html")

@app.route('/covidfaq')
@login_required
def covidfaq():
    return render_template("covidFaq.html")

@app.route('/covidhelp')
@login_required
def covidhelp():
    return render_template("covidHelp.html")

@app.route('/Symptom_Query')
@login_required
def symptomquery():
    return render_template("symptomQuery.html")


if __name__ == "__main__":
    app.run(debug=True) # From stopping Flask initializing twice in the debug mode.

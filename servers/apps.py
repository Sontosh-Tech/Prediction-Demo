import pickle
import re
import numpy as np

import servers
from servers import nlp,tfIdf
from flask import Flask, render_template, request, jsonify,redirect
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
model=pickle.load(open('ML_model.pkl', 'rb'))


@app.route('/')
def page():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def pred():
    if request.method == 'POST':
        if 'regression' in request.form.get('action'):
            features=[float(x) for x in request.form.values() if x!='regression']
            features_final=[np.array(features)]
            prediction=model.predict(features_final)
            prediction=round(prediction[0],2)
            return render_template('index.html',prediction='The predicted CO2 Emission will be {}'.format(prediction))
        else:
            return redirect('/NLP-predict-tfidf')


@app.route('/predict-api')
def prd_api():
    data=request.get_json(force=True)
    prediction=model.predict(np.array(list(data.values())))
    prediction = round(prediction[0], 2)
    return jsonify(prediction)

@app.route('/NLP')
def nlp():
    return render_template('NLP.html')

@app.route('/NLP-predict-<string:id>',methods=['GET','POST'])
def nlp_pred(id):
    if request.method=='POST':
        if id=='tfidf':
            model_nlp = pickle.load(open('nlp_tfidf.pkl', 'rb'))
            X_input=request.form.get('content')
            X=re.sub('[^a-zA-Z]',' ',str(X_input)).lower().split()
            wl=WordNetLemmatizer()
            X=[wl.lemmatize(word) for word in X if not word in set(stopwords.words('english'))]
            X = ' '.join(X)
            x= tfIdf.cv.transform([X,]).toarray()
        elif id=='bow':
            model_nlp = pickle.load(open('nlp_bow.pkl', 'rb'))
            X_input = request.form.get('content')
            X = re.sub('[^a-zA-Z]', ' ', str(X_input)).lower().split()
            ps = PorterStemmer()
            X = [ps.stem(word) for word in X if not word in set(stopwords.words('english'))]
            X = ' '.join(X)
            x = servers.nlp.cv.transform([X, ]).toarray()
        prediction = model_nlp.predict(x)
        prediction = int(prediction[0])
        return render_template('NLP.html', prediction=prediction,id=id)
    return render_template('NLP.html',id=id)


if __name__ == '__main__':
    app.run(debug=True)
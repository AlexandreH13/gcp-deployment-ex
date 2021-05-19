import pickle
import numpy as np
from flask import Flask, request

app = Flask(__name__)

model = pickle.load(open('iris_model_v1.pkl', 'rb')) ## Carrega o modelo

@app.route('/api_predict', methods=['GET', 'POST'])
def api_predict():
    if request.method == 'GET':
        return "É necessário um POST"
    elif request.method == 'POST':
        '''Input que nos vamos fornecer'''
        data = request.get_json()
        slength = data['sepal_length']
        swidth = data['sepal_width']
        plength = data['petal_length']
        pwidth = data['petal_width']

        ''' Passamos o input para um formato nprray pra o predict '''
        input1 = np.array([[slength, swidth, plength, pwidth]])

        pred = model.predict(input1)

        return str(pred)

if __name__=='__main':
    app.run()
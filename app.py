from flask import Flask,Response,render_template
from flask import request
import pandas as pd
from werkzeug.utils import secure_filename
import json
import pickle
import os
path = os.getcwd()
#template_path=path+'/templates'
port = int(os.getenv("PORT", 3000))
upload_folder = path
ALLOWED_EXTENSIONS = set(['pkl','txt'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = upload_folder
@app.route('/upload')
def upload():
   return render_template('upload.html')

@app.route('/uploader', methods = ['GET','POST'])
def upload_file():
    try:
        f = request.files['file']
        f.save(secure_filename(f.filename))
        print('file uploaded successfully')
        return 'file uploaded successfully'
    except Exception as e:
        print(e)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        linReg1 = pickle.load(open('predict1.pkl', 'rb'))
        linReg2 = pickle.load(open('predict2.pkl', 'rb'))
        linReg3 = pickle.load(open('predict3.pkl', 'rb'))
        linReg4 = pickle.load(open('predict4.pkl', 'rb'))
        linReg5 = pickle.load(open('predict5.pkl', 'rb'))

        column_data = pd.read_csv(path+'/columns.csv')
        column_1 = (column_data.columns[0])
        print("column 1",column_1) #size
        column_2 = (column_data.columns[1])
        print("column 2", column_2) #bedrooms
        column_3 = (column_data.columns[2])
        print("column 3", column_3) #age
        column_4 = (column_data.columns[3])
        print("column 4", column_4) #bathrooms
        column_5 = (column_data.columns[4])
        print("column 5", column_5) #Price

        req_body = request.get_json(force=True)
        print(req_body)

        # For Size
        if req_body[column_1] == '':
            param1 = req_body[column_2]
            param2 = req_body[column_3]
            param3 = req_body[column_4]
            param4 = req_body[column_5]
            pred = linReg1.predict([[param1, param2, param3, param4]])
            result = pred
            msg = {
                "Predicted value is": "%s" % (result)
            }
            resp = Response(response=json.dumps(msg),
                            status=200,
                            mimetype="application/json")
            return resp
        # For Bedrooms
        if req_body[column_2] == '':
            param1 = req_body[column_1]
            param2 = req_body[column_3]
            param3 = req_body[column_4]
            param4 = req_body[column_5]
            pred = linReg2.predict([[param1, param2, param3, param4]])
            result = pred
            msg = {
                "Predicted value is": "%s" % (result)
            }
            resp = Response(response=json.dumps(msg),
                            status=200,
                            mimetype="application/json")
            return resp
        # For Age
        if req_body[column_3] == '':
            param1 = req_body[column_1]
            param2 = req_body[column_2]
            param3 = req_body[column_4]
            param4 = req_body[column_5]
            pred = linReg3.predict([[param1, param2, param3, param4]])
            result = pred
            msg = {
                "Predicted value is": "%s" % (result)
            }
            resp = Response(response=json.dumps(msg),
                            status=200,
                            mimetype="application/json")
            return resp
        # For Bathrooms
        if req_body[column_4] == '':
            param1 = req_body[column_1]
            param2 = req_body[column_2]
            param3 = req_body[column_3]
            param4 = req_body[column_5]
            pred = linReg4.predict([[param1, param2, param3, param4]])
            result = pred
            msg = {
                "Predicted value is ": "%s" % (result)
            }
            resp = Response(response=json.dumps(msg),
                            status=200,
                            mimetype="application/json")
            return resp
        # For Price
        if req_body[column_5] == '':
            param1 = req_body[column_1]
            param2 = req_body[column_2]
            param3 = req_body[column_3]
            param4 = req_body[column_4]
            pred = linReg5.predict([[param1, param2, param3, param4]])
            result = pred
            msg = {
                "Predicted value is ": "%s" % (result)
            }
            resp = Response(response=json.dumps(msg),
                            status=200,
                            mimetype="application/json")
            return resp
    except Exception as e:
        print(e)
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=port)
from flask import Flask, jsonify, make_response, request
from flask_cors import CORS
import os, sys, time, warnings
from datetime import datetime, timedelta
import pandas as pd, numpy as np
import torch
from detect import detect
# 경고 무시
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# 날짜 설정
today = '20240415'

# 경로 설정
# src_path = '/home/tako/eoks/lh/lh_dev2/src'
# # 경로 추가
# if src_path not in sys.path:
#     sys.path.append(src_path)

# GPU 설정
# torch.cuda.set_device(1)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'source')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def tensor_to_serializable(data):
    """Converts nested tensors in a structure to serializable formats."""
    if isinstance(data, torch.Tensor):
        return data.tolist()  # convert tensor to list
    elif isinstance(data, (list, tuple)):
        return [tensor_to_serializable(item) for item in data]
    elif isinstance(data, dict):
        return {key: tensor_to_serializable(value) for key, value in data.items()}
    return data

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        return jsonify({'file_name': filename, 'file_path': file_path}), 200

@app.route('/predict', methods=['POST'])
def pred_img():
    file_name = request.form.get('file_name')
    file_path = request.form.get('file_path')
    lines = detect(saved_img_name=file_name, saved_txt_name=file_name+'.txt', source=file_path, device="cpu") #device=0
    # Convert tensors to serializable formats
    serializable_lines = tensor_to_serializable(lines)
    
    # Convert to JSON response
    return jsonify(serializable_lines)

@app.route('/text/predict', methods=['POST'])
def pred_text():
    # text 변수
    text = request.form.get('text')
    num = request.form.get('num')
    # Convert to JSON response
    tmp = {
        'text': text,
        'num': num
    }
    return tmp

@app.route('/', methods=['GET'])
def test_response():
    """Return a sample JSON response."""
    sample_response = {
        "items": [
            { "id": 1, "name": "Apples",  "price": "$2" },
            { "id": 2, "name": "Peaches", "price": "$5" }
        ]
    }
    # JSONify response
    response = make_response(jsonify(sample_response))

    # Add Access-Control-Allow-Origin header to allow cross-site request
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'

    # Mozilla provides good references for Access Control at:
    # https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS
    # https://developer.mozilla.org/en-US/docs/Web/HTTP/Server-Side_Access_Control

    return response

if __name__ == "__main__":
    app.run()
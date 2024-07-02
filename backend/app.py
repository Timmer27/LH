from flask import Flask, jsonify, make_response, request
from flask_cors import CORS
import os, sys, time, warnings
from datetime import datetime, timedelta
import pandas as pd, numpy as np
import torch
from detect import detect
import pandas as pd  # 데이터 처리
from torch.utils.data import Dataset, DataLoader  # 데이터 로딩
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # 트랜스포머 모델
import torch.nn.functional as F

# 사용자 정의 모듈 로드
from common import *
from func_collection import *

# 경고 무시
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# 경로 설정
# src_path = '/home/tako/eoks/lh/lh_dev2/src'
# # 경로 추가
# if src_path not in sys.path:
#     sys.path.append(src_path)

# GPU 설정
# torch.cuda.set_device(1)

# 데이터셋 클래스 정의
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

TEXT_MODEL_FILES = os.path.join(os.getcwd(), 'models_text')
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'source')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 텍스트 예측 함수 정의
def txt_prd_func(model, flw_cts, word_dict, tokenizer, labels_df, num:int=3):
    start_time = time.time()  # 시작 시간 저장

    # 입력 처리'../data/dic/불용어.txt'
    input_text = flw_cts
    pre_input_text = preprocess_text(TEXT_MODEL_FILES, 'stopwords.txt', input_text)
    stan_input_text = replace_words(pre_input_text, 1, word_dict)

    # 데이터 토큰화, 데이터셋 생성, 데이터 로더 생성
    test_encodings = tokenizer([stan_input_text], truncation=True, padding=True)    
    test_dataset = TextDataset(test_encodings, [0])         
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 장치 설정
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    # 모델 평가
    model.to(device)
    model.eval()        
    for batch in test_loader:  
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)

    # 결과 계산
    probabilities = F.softmax(outputs.logits, dim=1)
    prob, indices = torch.topk(probabilities, num)

    # 응답 생성
    result = []
    for i, label_index in enumerate(indices.tolist()[0][:num]):
        result.append({
            "tp_cd": labels_df['tp_cd'].loc[label_index],
            "tp_nm": labels_df['labels'].loc[label_index],
            "tp_prb": round(prob.tolist()[0][i], 4)
        })

    def _ordinal_suffix(n):
        if 11 <= n % 100 <= 13:
            return 'th'
        else:
            return {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')        

    # Rename keys for the first three results
    key_mapping = {i: f"{_ordinal_suffix(i+1)}_tp" for i in range(num)}
    # key_mapping = {0: "fir", 1: "sec", 2: "thi"}
    rspns = {"result": [{f"{key_mapping[i]}_tp_cd": item["tp_cd"], f"{key_mapping[i]}_tp_nm": item["tp_nm"], f"{key_mapping[i]}_tp_prb": item["tp_prb"]} for i, item in enumerate(result)], "status": "ok"}

    end_time = time.time()  # 종료 시간 저장
    print("응답시간: ", end_time - start_time)  # 현재시각 - 시작시간 = 실행 시간

    return rspns


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
    # num = request.form.get('num')

    # 표준 단어 세트 로드 및 대표단어 변경
    standard_word_set = pd_read_json(os.path.join(TEXT_MODEL_FILES, 'TB_MMA_CNV_DIC_M.json'))
    word_dict = dict(zip(standard_word_set['INP_DTLS'], standard_word_set['RPSN_DTLS']))

    # 레이블 데이터 로드
    labels_df = pd.read_excel(os.path.join(TEXT_MODEL_FILES, 'labels_task1.xlsx'))
    labels_df = labels_df.drop('Unnamed: 0', axis=1)

    # GPU 설정
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device('cuda:0')
        print("Using GPU:", torch.cuda.get_device_name(0))
        tokenizer = torch.load(os.path.join(TEXT_MODEL_FILES, 'dev_dev_lh_text_clf_tok_20240625_thi.pt'))
        model = torch.load(os.path.join(TEXT_MODEL_FILES, 'dev_dev_lh_text_clf_20240625_thi.pt'))
    else:
        device = torch.device('cpu')
        tokenizer = torch.load(os.path.join(TEXT_MODEL_FILES, 'dev_dev_lh_text_clf_tok_20240625_thi.pt'))
        model = torch.load(os.path.join(TEXT_MODEL_FILES, 'dev_dev_lh_text_clf_20240625_thi.pt'), map_location=torch.device('cpu'))    
        print("Using CPU")

    data = txt_prd_func(model, text, word_dict, tokenizer, labels_df)
    # Convert int64 to int
    for result in data['result']:
        for key, value in result.items():
            if isinstance(value, np.int64):
                result[key] = int(value)

    return jsonify(data)

@app.route('/test', methods=['GET'])
def test_response2():
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
    app.run(debug=True)
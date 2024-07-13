from flask import Blueprint, jsonify, request
import pandas as pd
import numpy as np
import torch, os, sys
from config import Config
import pickle
from utils.helpers import pd_read_json, pd_read_file
from services.image_service import detect
from services.text_service import txt_prd_func
from services.schedule_service import schdl_func 
from services.automation_service import atmtd_rprt_func
from utils.helpers import tensor_to_serializable

predict_bp = Blueprint('predict', __name__)
STANDARD_WORD_SET = pd_read_json(Config.STANDARD_WORD_SET_PATH)

@predict_bp.route('/predict/image', methods=['POST'])
def pred_img():
    file_name = request.form.get('file_name')
    file_path = request.form.get('file_path')
    lines = detect(saved_img_name=file_name, saved_txt_name=file_name+'.txt', source=file_path, device="cpu") #device=0
    # Convert tensors to serializable formats
    serializable_lines = tensor_to_serializable(lines)
    
    # Convert to JSON response
    return jsonify(serializable_lines)

@predict_bp.route('/predict/text', methods=['POST'])
def pred_text():
# text 변수
    text = request.form.get('text')
    flwDtlSn = request.form.get('text')
    flwDsCd = request.form.get('text')
    # num = request.form.get('num')
    
    # 표준 단어 세트 로드 및 대표단어 변경
    word_dict = dict(zip(STANDARD_WORD_SET['INP_DTLS'], STANDARD_WORD_SET['RPSN_DTLS']))

    # 레이블 데이터 로드
    labels_df = pd.read_excel(os.path.join(Config.TEXT_MODEL_FILES, 'labels_task1.xlsx'))
    labels_df = labels_df.drop('Unnamed: 0', axis=1)

    # GPU 설정
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")

    tokenizer = torch.load(os.path.join(Config.TEXT_MODEL_FILES, 'dev_dev_lh_text_clf_tok_20240625_thi.pt'), map_location=device)
    model = torch.load(os.path.join(Config.TEXT_MODEL_FILES, 'dev_dev_lh_text_clf_20240625_thi.pt'), map_location=device)


    data = txt_prd_func(model, text, word_dict, tokenizer, labels_df)
    # Convert int64 to int
    for result in data['result']:
        for key, value in result.items():
            if isinstance(value, np.int64):
                result[key] = int(value)

    return jsonify(data)


@predict_bp.route('/predict/schedule', methods=['POST'])
def pred_schedule():
    inpt = request.get_json()
    # 학습 및 검증 데이터 로드
    train_contents =  pd.read_csv(os.path.join(Config.SCHEDULE_MODEL_FILES, 'dev_pre_2022년 하자 및 유지보수접수건(1~4분기)_train_20240529.csv'), encoding='cp949')
    # 데이터 결측치 제거
    train_contents['preprocessed_con'].fillna(' ', inplace=True)

    # 레이블 데이터 로드
    labels_df = pd.read_excel(os.path.join(Config.SCHEDULE_MODEL_FILES, 'labels_task1.xlsx'))
    labels_df = labels_df.drop('Unnamed: 0', axis=1)

    # 표준 단어 세트 로드 및 대표단어 변경
    word_dict = dict(zip(STANDARD_WORD_SET['INP_DTLS'], STANDARD_WORD_SET['RPSN_DTLS']))

    # 데이터 로드
    with open(os.path.join(Config.SCHEDULE_MODEL_FILES, 'preprocessed_corpus.pkl'), 'rb') as f:
        loaded_preprocessed_corpus = pickle.load(f)
        
    # GPU 설정
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    print("Using GPU:", torch.cuda.get_device_name(0))
    # 학습된 모델 로드
    model = torch.load(os.path.join(Config.SCHEDULE_MODEL_FILES, 'dev_BERT_TSDAE_MODEL_20240529.pt'), map_location=device)
    # 임베딩 벡터 로드
    train_embeddings = torch.load(os.path.join(Config.SCHEDULE_MODEL_FILES, 'dev_corpus_embedded_20240529.pt'),  map_location=device).float()

    rspns= schdl_func(inpt, model, word_dict, loaded_preprocessed_corpus, labels_df, train_contents, train_embeddings)

    print('rspns', rspns)
    return jsonify(rspns)


@predict_bp.route('/predict/automated', methods=['POST'])
def pred_automated():
    inpt = request.get_json()
    # 테스트 데이터 로드
    train_contents = pd_read_file(os.path.join(Config.AUTOMATED_MODEL_FILES, 'dev_pre_refair_train_20240531.csv'))
    test_contents = pd_read_file(os.path.join(Config.AUTOMATED_MODEL_FILES, 'dev_pre_refair_test_20240531.csv'))
    dup_train_list = train_contents.drop_duplicates(subset=['접수번호']).reset_index(drop=True)
    dup_test_list = test_contents.drop_duplicates(subset=['접수번호']).reset_index(drop=True)
    train_list = dup_train_list['preprocessed_con'].tolist()
    test_list = dup_test_list['preprocessed_con'].tolist()

    # 세부내역서 로드
    breakdown_cost_list = pd.read_excel(os.path.join(Config.AUTOMATED_MODEL_FILES, '서울북부권_세부내역서.xlsx'))
    breakdown_cost_list = breakdown_cost_list[~breakdown_cost_list['연번'].isna()]
    breakdown_cost_list.fillna('', inplace=True)

    # 결측치 처리
    test_contents.fillna('', inplace=True)

    # 표준 단어 세트 로드 및 대표단어 변경
    word_dict = dict(zip(STANDARD_WORD_SET['INP_DTLS'], STANDARD_WORD_SET['RPSN_DTLS']))

    # 데이터의 길이를 출력 Length of test_contents :  87
    print('Length of test_contents : ',len(test_contents)) 
    print("Using GPU:", torch.cuda.get_device_name(0))
    # GPU 설정
    device = torch.device('cuda:0')
    # 학습된 모델 로드
    model = torch.load(os.path.join(Config.AUTOMATED_MODEL_FILES, 'dev_BERT_TSDAE_MODEL_20240531.pt'))
    # 학습된 임베딩 벡터 로드
    train_embeddings = torch.load(os.path.join(Config.AUTOMATED_MODEL_FILES, 'dev_corpus_embedded_20240531.pt'))

    # 데이터 로드 .
    with open(os.path.join(Config.AUTOMATED_MODEL_FILES, 'preprocessed_corpus.pkl'), 'rb') as f:
        loaded_preprocessed_corpus = pickle.load(f)
        preprocessed_corpus = loaded_preprocessed_corpus

    data = atmtd_rprt_func(inpt, word_dict, model, train_list, preprocessed_corpus, train_contents, dup_train_list, breakdown_cost_list)
    return jsonify(data)
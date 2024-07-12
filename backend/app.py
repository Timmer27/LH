from flask import Flask, jsonify, make_response, request, send_from_directory
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
import pickle

# 사용자 정의 모듈 로드
from common import *
from func_collection import *

# 경고 무시
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# app.config['RESULTS_FOLDER'] = os.path.join(os.getcwd(), 'results')

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

TEXT_MODEL_FILES = os.path.join(os.getcwd(), 'model_text')
SCHEDULE_MODEL_FILES = os.path.join(os.getcwd(), 'model_schedule')
AUTOMATED_MODEL_FILES = os.path.join(os.getcwd(), 'model_automated')
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'source')
OUTPUT_FOLDER = os.path.join(os.getcwd(), 'results')
STANDARD_WORD_SET = pd_read_json(os.path.join(os.getcwd(), 'TB_MMA_CNV_DIC_M.json'))
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 텍스트 예측 함수 정의
def txt_prd_func(model, flw_cts, word_dict, tokenizer, labels_df, flwDtlSn:str="01234567", flwDsCd:str="입주사사전방문(웰컴데이)", num:int=3):
    
    """
    응답 예시
        {
            "status": "success",
            "message": "Data processed successfully",
            "defect_info_detail": {
                "flwDtlSn": 01234567,
                "flwDsCd": "입주사사전방문(웰컴데이)"},
        "result": [{"fir_tp_cd": "01", "fir_tp_nm": "개폐불량", "fir_tp_prb": "0.9"},
                    {"sec_tp_cd": "02", "sec_tp_nm": "작동불량", "sec_tp_prb": "0.8"},
                    {"thi_tp_cd": "03", "thi_tp_nm": "마감불량", "thi_tp_prb": "0.7"}]
        }
    """
    start_time = time.time()  # 시작 시간 저장

    # 입력 처리'../data/dic/불용어.txt'
    input_text = flw_cts
    pre_input_text = preprocess_text(os.getcwd(), 'stopwords.txt', input_text)
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
    rspns = {
        "status": "success",
        "message": "Data processed successfully",
        "defect_info_detail": {
            "flwDtlSn": flwDtlSn,
            "flwDsCd": flwDsCd
            },
        "result": 
            [{f"{key_mapping[i]}_tp_cd": item["tp_cd"],
                f"{key_mapping[i]}_tp_nm": item["tp_nm"],
                f"{key_mapping[i]}_tp_prb": item["tp_prb"]}
            for i, item in enumerate(result)],
        }

    end_time = time.time()  # 종료 시간 저장
    print("응답시간: ", end_time - start_time)  # 현재시각 - 시작시간 = 실행 시간

    return rspns

# class 

def schdl_func(inpt, model, word_dict, loaded_preprocessed_corpus, labels_df, train_contents, train_embeddings):
    start_time = time.time()  # 시작 시간 저장
    top_k = 3  # 상위 k개의 결과를 보기 위한 설정

    print('inpt', inpt)
    flw_ds_cd = inpt["defect_info_detail"]['flwDsCd']
    flw_acp_sn = inpt['defect_info_detail']['flwDtlSn']
    spce_cd = inpt["inp"]['spce_cd']
    comp = inpt["inp"]['comp']
    tp_nm = inpt["inp"]['tp_nm']
    flw_cts = inpt["inp"]['flw_cts']
    
    pre_flw_cts = preprocess_text(os.getcwd(), 'stopwords.txt', flw_cts)
    stan_flw_cts = replace_words(pre_flw_cts, 1, word_dict)

    # 유사도 계산 방법 설정
    similarity_methods = ['cosine', 'euclidean', 'manhattan', 'jaccard', 'overlap']
    # similarity_methods = ['cosine', 'euclidean']

    # 결과 저장을 위한 데이터프레임 생성
    columns = ['하자접수일련번호', 'AI_모델_예측_하자유형', '보수지시일','보수예정일자','보수완료보고일자','보수완료일자']
    result_sim = pd.DataFrame(columns=columns)

    # 임베딩 수행
    query_embedding = model.encode(stan_flw_cts, convert_to_tensor=True).cpu()

    # 열 값 입력
    column_values = {
        '하자구분': flw_ds_cd,
        '(신)공간': spce_cd,
        '(신)부위자재': comp,
        '(신)하자유형': tp_nm,
    }

    # 학습 데이터 필터링
    filtered_train, filtered_tensor, idx = filter_data(train_contents, train_embeddings, column_values, top_k)
    preprocessed_corpus = [loaded_preprocessed_corpus[i] for i in idx]

    # 유사도 계산 및 결과 저장
    for method in similarity_methods:
        top_results, score = cal_sim_results(stan_flw_cts, query_embedding, filtered_tensor, top_k, method, preprocessed_corpus)        
        for j in range(top_k):
            result_row = [flw_acp_sn,
                        tp_nm,
                        filtered_train.loc[top_results.tolist()[j]]['보수지시일'],
                        filtered_train.loc[top_results.tolist()[j]]['보수예정일자'],
                        filtered_train.loc[top_results.tolist()[j]]['보수완료보고일자'],
                        filtered_train.loc[top_results.tolist()[j]]['보수완료일자']
                        ]
            result_sim.loc[len(result_sim)] = result_row   

    ## 스케줄링 기능 영역
    # 날짜 칼럼들을 datetime 형식으로 변환
    # 날짜 형식 변환 및 보수일정기간 계산
    result_sim['보수지시일'] = pd.to_datetime(result_sim['보수지시일'], errors='coerce')
    result_sim['보수예정일자'] = pd.to_datetime(result_sim['보수예정일자'], errors='coerce')
    result_sim['보수완료보고일자'] = pd.to_datetime(result_sim['보수완료보고일자'], errors='coerce')
    result_sim['보수완료일자'] = pd.to_datetime(result_sim['보수완료일자'], errors='coerce')
    result_sim['보수소요일'] = result_sim.apply(calculate_duration, axis=1).dt.days.fillna(0).astype(int)
    result_sim['투입인력수'] = result_sim['보수소요일'].apply(assign_workers)
    result_sim['예상보수소요일'] = np.repeat(result_sim.groupby('하자접수일련번호')['보수소요일'].mean().values, len(similarity_methods) * top_k).astype(int).tolist()
    result_sim['예상투입인력수'] = np.repeat(result_sim.groupby('하자접수일련번호')['투입인력수'].mean().values, len(similarity_methods) * top_k).astype(int).tolist()

    # 결과저장
    out_result_sim = result_sim[['하자접수일련번호', 'AI_모델_예측_하자유형', '예상보수소요일', '예상투입인력수']]
    out_result_sim.fillna("", inplace=True)
    out_result_sim.drop_duplicates(inplace=True)
    out_result_dict = out_result_sim.loc[0].to_dict()

    # 결과양식 작성
    rspns = {
        "status": "success",
        "message": "Data processed successfully",
        "defect_info_detail": {
                            "flwDtlSn": flw_acp_sn,
                            "flwDsCd": flw_ds_cd},
        "result": {"prd_tp_cd": str(labels_df[labels_df['labels'] == out_result_dict['AI_모델_예측_하자유형']]['tp_cd'].values[0]),
                "prd_tp_nm": out_result_dict['AI_모델_예측_하자유형'],
                "prd_drtn": out_result_dict['예상보수소요일'],
                "inp_wrk_cnt": out_result_dict['예상투입인력수'],
                "vst_fir_dt": (datetime.today() + timedelta(days=1)).strftime('%Y%m%d'),
                "vst_sec_dt": (datetime.today() + timedelta(days=8)).strftime('%Y%m%d'),
                "vst_thi_dt": (datetime.today() + timedelta(days=15)).strftime('%Y%m%d')}
            }
    end_time = time.time()  # 종료 시간 저장
    print("응답시간: ", end_time - start_time)  # 현재시각 - 시작시간 = 실행 시간
    return rspns

def atmtd_rprt_func(inpt, word_dict, model, train_list, preprocessed_corpus, train_contents, dup_train_list, breakdown_cost_list):
    start_time = time.time()  # 시작 시간 저장
    # 상위 k개의 유사도를 계산
    top_k = 3
    n_query = 1

    print('inpt',inpt)

    flwDtlSn = inpt['defect_info_detail']['flwDtlSn']
    flwDsCd = inpt['defect_info_detail']['flwDsCd']
    flw_cts = inpt['inp']['flw_cts']

    pre_flw_cts = preprocess_text(os.getcwd(), 'stopwords.txt', flw_cts)
    stan_flw_cts = replace_words(pre_flw_cts, 1, word_dict)

    # 결과를 저장할 데이터프레임을 초기화.
    result_sim = pd.DataFrame(columns=['접수번호','접수내용', '유사도방법', 'top_k', 'k_score','공종', '연번', '권역', '구분', '단지', '동', '호',
                                    '하자번호', '하자내용', '타입', '코드', '명칭', '규격', '단위', '계약자재비', '계약노무비', '계약경비',
                                    '계약계', '산출내역', '수량','공사자재비', '공사노무비', '공사경비', '공사계', '공사자재비소계', 
                                    '공사노무비소계', '공사경비소계', '공사소계' ])
                                    

    # 유사도 계산 방법을 설정
    # similarity_methods = ['cosine', 'euclidean', 'manhattan', 'jaccard', 'overlap']
    similarity_methods = ['cosine', 'jaccard', 'overlap']

    # 형태소 분석기를 초기화하고, 사전에 단어를 추가
    twitter = initialize_morpheme_analyzer(STANDARD_WORD_SET)

    # 접수내용 리스트를 임베딩합니다.
    corpus_embeddings = model.encode(train_list, convert_to_tensor=True).cpu()  

    # 테스트 콘텐츠를 반복하며 각 쿼리에 대해 유사도를 계산 
    query_embedding = model.encode(stan_flw_cts, convert_to_tensor=True).cpu()    

    # 각 유사도 방법에 대해 상위 결과를 계산하고 결과를 저장
    for method in similarity_methods:
        top_results, score = cal_sim_results(stan_flw_cts, query_embedding, corpus_embeddings, top_k, method, preprocessed_corpus) 
        for j in range(top_k):
            rows = train_contents[train_contents['접수번호'] == dup_train_list['접수번호'][top_results.tolist()[j]]]
            for _, row in rows.iterrows():
                result_sim.loc[len(result_sim)] = [flwDtlSn,
                                                    flw_cts,
                                                    method, 
                                                    j+1,  
                                                    round(score.tolist()[j],4),
                                                    row['공종'],
                                                    row['연번'], 
                                                    row['권역'],
                                                    row['구분'],
                                                    row['단지'],
                                                    row['동'],
                                                    row['호'],
                                                    row['접수번호'],
                                                    row['접수내용'],
                                                    row['타입'],
                                                    row['코드'],
                                                    row['명칭'],
                                                    row['규격'],
                                                    row['단위'], 
                                                    0, 0, 0, 0, 0, row['수량'],
                                                    0, 0, 0, 0, 0, 0, 0, 0
                                                    ]

    # columns_to_consider = ['명칭', '규격', '수량']
    columns_to_consider = ['명칭', '규격']
    # 최종 데이터프레임을 초기화
    final_df = pd.DataFrame()

    for i in range(len(result_sim['접수번호'].unique().tolist())):
        for method in similarity_methods:
            # 필터링된 데이터프레임을 생성
            filtered_result_sim = result_sim[(result_sim['접수번호'] == result_sim['접수번호'].unique().tolist()[i]) & (result_sim['유사도방법'] == method)]
            
            # 필터링된 데이터프레임을 정렬하고 중복을 제거
            df = sort_and_remove_duplicates(filtered_result_sim, columns_to_consider)
            
            # 최종 데이터프레임에 추가
            final_df = pd.concat([final_df, df], ignore_index=True)

    ##
    final_result1 = final_df[['접수번호', '접수내용', '유사도방법', 'k_score', '공종','단지','동','호','하자내용','명칭','규격','단위', '수량']]
    new_columns = ['계약자재비', '계약노무비', '계약경비', '계약계', '공사자재비', '공사노무비', '공사경비', '공사계']
    final_result2 = final_result1.copy()

    for col in new_columns:
        final_result2[col] = 0

    final_result2 = final_result2[['접수번호', '접수내용', '유사도방법', 'k_score', '공종','단지','동','호','하자내용','명칭','규격','단위', \
                                '계약자재비', '계약노무비', '계약경비', '계약계', '수량', '공사자재비', '공사노무비', '공사경비', '공사계']]

    # 접수내용에서 명칭과 규격 단위에 맞는 세부내역의 금액을 각각 재료비는 계약자재비에 노무비는 계약노무비에 그리고 경비는 계약경비에 값을 입력
    df1 = final_result2
    df2 = breakdown_cost_list.drop(['연번'], axis=1).reset_index(drop=True).rename(lambda x: x + 1)
    for index, row in df1.iterrows():
        matched_row = df2[(df2['명칭'] == row['명칭']) & (df2['규격'] == row['규격']) & (df2['단위'] == row['단위'])]
        if not matched_row.empty:
            df1.at[index, '계약자재비'] = matched_row['재료비'].values[0]
            df1.at[index, '계약노무비'] = matched_row['노무비'].values[0]
            df1.at[index, '계약경비'] = matched_row['경비'].values[0]

    # 수량에 따라서 계약자재비 * 수량은 공사자재비로 노무비 * 수량은 공사노무비로 경비 * 수량은 공사경비로 값을 넣음
    df1['공사자재비'] = df1['계약자재비'] * df1['수량']
    df1['공사노무비'] = df1['계약노무비'] * df1['수량']
    df1['공사경비'] = df1['계약경비'] * df1['수량']

    # 계약계, 공사계 산출
    df1['계약계'] = df1['계약자재비'] + df1['계약노무비']+ df1['계약경비']
    df1['공사계'] = df1['공사자재비'] + df1['공사노무비']+ df1['공사경비']

    # 분류코드 매핑 딕셔너리 생성
    classification_dict = dict(zip(df2['명칭'] + df2['규격'], df2['분류']))

    # 명칭과 규격에 맞는 분류코드로 칼럼 추가
    df1['분류코드'] = (df1['명칭'] + df1['규격']).map(classification_dict)
    # final_result1.reset_index(drop=True).rename(lambda x: x + 1)

    ## 일위대가 연동 결과:
    df1 = df1.reset_index(drop=True).rename(lambda x: x + 1)
    breakdown_cost_result = df1[['접수번호', '접수내용','명칭', '규격', '단위', '계약자재비', '계약노무비', '계약경비', '계약계', '수량', '공사자재비', '공사노무비', '공사경비', '공사계']]

    # 계약 비용으로 하는 경우
    vis_df1 = df1[['접수번호','분류코드', '명칭','규격', '단위', '수량', '계약자재비', '계약노무비', '계약경비', '계약계']]
    vis_df1.columns = ['접수번호','일위대가코드', '일위대가명','규격', '단위', '수량', '재료비', '노무비', '경비', '합계']

    # # 계약 비용 * 수량으로 하는 경우
    # vis_df1 = df1[['접수번호','분류코드', '명칭','규격', '단위', '수량', '공사자재비', '공사노무비', '공사경비', '공사계']]
    # vis_df1.columns = ['접수번호','일위대가코드', '일위대가명','규격', '단위', '수량', '재료비', '노무비', '경비', '합계']
    
    vis_df1_sorted = vis_df1.drop_duplicates().sort_values(by=['일위대가명', '수량'], ascending=False).reset_index(drop=True)
    vis_df1_api_format = vis_df1_sorted.apply(convert_to_api_format, axis=1).tolist()

    rspns = {
            "status": "success",
            "message": "Data processed successfully",
            "defect_info_detail": {
                                "flwDtlSn": flwDtlSn,
                                "flwDsCd": flwDsCd
                                },
            "result": {
                    "하자접수정보" : {"hno": "추후연동예정", "bldg_no": "추후연동예정", "hs_no": "추후연동예정"},

            "하자처리내역": {
                            "spce_cd": "추후연동예정", "부위자재": "추후연동예정", "ai_prd_tp_nm": "추후연동예정",
                            "flw_cts": flw_cts,
                            "cts": "추후연동예정",
                            "flw_rgs_dt": "추후연동예정", "vst_fix_dt": "추후연동예정", "rpr_cmpl_dt": "추후연동예정",
                            "rpr_drtn" : "추후연동예정", "inp_wrk_cnt": "추후연동예정", "frm_cd": "추후연동예정", "icpr_empno": "추후연동예정"
                            },

            "일위대가": vis_df1_api_format}
            }
    
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
        return jsonify({'file_name': filename, 'file_path': file_path, 'output_path': OUTPUT_FOLDER}), 200

@app.route('/image/<filename>')
def get_image(filename):
    image_directory = os.path.join(app.root_path, 'results')
    return send_from_directory(image_directory, filename)

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
    flwDtlSn = request.form.get('text')
    flwDsCd = request.form.get('text')
    # num = request.form.get('num')

    # 표준 단어 세트 로드 및 대표단어 변경
    word_dict = dict(zip(STANDARD_WORD_SET['INP_DTLS'], STANDARD_WORD_SET['RPSN_DTLS']))

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

@app.route('/schedule/predict', methods=['POST'])
def pred_schedule():
    inpt = request.get_json()
    # print('data', inpt['inp']['spce_cd'])
    # flw_dtl_sn, flw_df_cd = inpt['defect_info_detail']['flw_dtl_sn'], inpt['defect_info_detail']['flw_df_cd']
    # spce_cd, comp, tp_nm, flw_cts = inpt['inp']['spce_cd'], inpt['inp']['comp'], inpt['inp']['tp_nm'], inpt['inp']['flw_cts']

    # print('etsts', flw_dtl_sn, flw_df_cd, spce_cd, comp, tp_nm, flw_cts)
    # 학습 및 검증 데이터 로드
    train_contents =  pd.read_csv(os.path.join(SCHEDULE_MODEL_FILES, 'dev_pre_2022년 하자 및 유지보수접수건(1~4분기)_train_20240529.csv'), encoding='cp949')
    # 데이터 결측치 제거
    train_contents['preprocessed_con'].fillna(' ', inplace=True)

    # 레이블 데이터 로드
    labels_df = pd.read_excel(os.path.join(SCHEDULE_MODEL_FILES, 'labels_task1.xlsx'))
    labels_df = labels_df.drop('Unnamed: 0', axis=1)

    # 표준 단어 세트 로드 및 대표단어 변경
    word_dict = dict(zip(STANDARD_WORD_SET['INP_DTLS'], STANDARD_WORD_SET['RPSN_DTLS']))

    # 데이터 로드
    with open(os.path.join(SCHEDULE_MODEL_FILES, 'preprocessed_corpus.pkl'), 'rb') as f:
        loaded_preprocessed_corpus = pickle.load(f)
        
    # GPU 설정
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    print("Using GPU:", torch.cuda.get_device_name(0))
    # 학습된 모델 로드
    model = torch.load(os.path.join(SCHEDULE_MODEL_FILES, 'dev_BERT_TSDAE_MODEL_20240529.pt'), map_location=device)
    # 임베딩 벡터 로드
    train_embeddings = torch.load(os.path.join(SCHEDULE_MODEL_FILES, 'dev_corpus_embedded_20240529.pt'),  map_location=device).float()

    rspns= schdl_func(inpt, model, word_dict, loaded_preprocessed_corpus, labels_df, train_contents, train_embeddings)
    # rspns = {
    #     'status': 'success',
    #     'message': 'Data processed successfully',
    #     'defect_info_detail': {
    #             'flwDtlSn': 1234567,
    #             'flwDsCd': '입주사사전방문(웰컴데이)'},
    #     'result': {'prd_tp_cd': '84',
    #             'prd_tp_nm': '흠집',
    #             'prd_drtn': 15,
    #             'inp_wrk_cnt': 7,
    #             'vst_fir_dt': '20240704',
    #             'vst_sec_dt': '20240711',
    #             'vst_thi_dt': '20240718'}
    # }
    print('rspns', rspns)
    return jsonify(rspns)


@app.route('/automated/predict', methods=['POST'])
def pred_automated():
    inpt = request.get_json()
    # input = {
    #     "defect_info_detail": {
    #         "flwDtlSn": '0123456',
    #         "flwDsCd": "입주사사전방문(웰컴데이)"},
    #         "araHdqCd": "001",
    #     "inp": {"flw_cts": "주방 상부장이 흔들려요"}
    # }

    # 테스트 데이터 로드
    train_contents = pd_read_file(os.path.join(AUTOMATED_MODEL_FILES, 'dev_pre_refair_train_20240531.csv'))
    test_contents = pd_read_file(os.path.join(AUTOMATED_MODEL_FILES, 'dev_pre_refair_test_20240531.csv'))
    dup_train_list = train_contents.drop_duplicates(subset=['접수번호']).reset_index(drop=True)
    dup_test_list = test_contents.drop_duplicates(subset=['접수번호']).reset_index(drop=True)
    train_list = dup_train_list['preprocessed_con'].tolist()
    test_list = dup_test_list['preprocessed_con'].tolist()

    # 세부내역서 로드
    breakdown_cost_list = pd.read_excel(os.path.join(AUTOMATED_MODEL_FILES, '서울북부권_세부내역서.xlsx'))
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
    model = torch.load(os.path.join(AUTOMATED_MODEL_FILES, 'dev_BERT_TSDAE_MODEL_20240531.pt'))
    # 학습된 임베딩 벡터 로드
    train_embeddings = torch.load(os.path.join(AUTOMATED_MODEL_FILES, 'dev_corpus_embedded_20240531.pt'))
    # if torch.cuda.is_available():
    #     torch.cuda.set_device(0)
    # else:
    #     device = torch.device('cpu')
    #     # 학습된 모델 로드
    #     model = torch.load(os.path.join(AUTOMATED_MODEL_FILES, 'dev_BERT_TSDAE_MODEL_20240531.pt'), map_location=torch.device('cpu'))    
    #     # 학습된 임베딩 벡터 로드
    #     train_embeddings = torch.load(os.path.join(AUTOMATED_MODEL_FILES, 'dev_corpus_embedded_20240531.pt'))
    #     print("Using CPU")

    # 데이터 로드 .
    with open(os.path.join(AUTOMATED_MODEL_FILES, 'preprocessed_corpus.pkl'), 'rb') as f:
        loaded_preprocessed_corpus = pickle.load(f)
        preprocessed_corpus = loaded_preprocessed_corpus


    data = atmtd_rprt_func(inpt, word_dict, model, train_list, preprocessed_corpus, train_contents, dup_train_list, breakdown_cost_list)
    print('data = ', data)
    # data = txt_prd_func(model, text, word_dict, tokenizer, labels_df)
    # # Convert int64 to int
    # for result in data['result']:
    #     for key, value in result.items():
    #         if isinstance(value, np.int64):
    #             result[key] = int(value)

    return jsonify(data)

@app.route('/gputest', methods=['GET'])
def test_response3():
    # Check if CUDA is available
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    torch.cuda.empty_cache()
    # GPU 설정
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    print("Using GPU:", torch.cuda.get_device_name(0))
    # 학습된 모델 로드
    model = torch.load(os.path.join(SCHEDULE_MODEL_FILES, 'dev_BERT_TSDAE_MODEL_20240529.pt'), map_location=device)
    # 임베딩 벡터 로드
    train_embeddings = torch.load(os.path.join(SCHEDULE_MODEL_FILES, 'dev_corpus_embedded_20240529.pt'),  map_location=device).float()

    return "hi test"

@app.route('/test', methods=['GET'])
def test_response2():
    """Return a sample JSON response."""
    sample_response = {
        "items": [
            { "id": 1, "name": "Appsssssles",  "price": "$2" },
            { "id": 2, "name": "Peaches", "price": "$5" }
        ]
    }    # GPU 설정
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    print("Using GPU:", torch.cuda.get_device_name(0))
    # 학습된 모델 로드
    model = torch.load(os.path.join(SCHEDULE_MODEL_FILES, 'dev_BERT_TSDAE_MODEL_20240529.pt'))
    # 임베딩 벡터 로드
    train_embeddings = torch.load(os.path.join(SCHEDULE_MODEL_FILES, 'dev_corpus_embedded_20240529.pt')).float()
    # JSONify response
    response = make_response(jsonify(sample_response))
    # GPU 설정
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    print("Using GPU:", torch.cuda.get_device_name(0))
    # 학습된 모델 로드
    model = torch.load(os.path.join(SCHEDULE_MODEL_FILES, 'dev_BERT_TSDAE_MODEL_20240529.pt'))
    # 임베딩 벡터 로드
    train_embeddings = torch.load(os.path.join(SCHEDULE_MODEL_FILES, 'dev_corpus_embedded_20240529.pt')).float()
    # Add Access-Control-Allow-Origin header to allow cross-site request
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'

    # Mozilla provides good references for Access Control at:
    # https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS
    # https://developer.mozilla.org/en-US/docs/Web/HTTP/Server-Side_Access_Control

    return response

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
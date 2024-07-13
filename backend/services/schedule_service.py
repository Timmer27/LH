import time
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from utils.preprocessing import preprocess_text, replace_words, filter_data, cal_sim_results, calculate_duration, assign_workers

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

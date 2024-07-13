import time
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from utils.preprocessing import initialize_morpheme_analyzer, preprocess_text, replace_words, sort_and_remove_duplicates, convert_to_api_format, cal_sim_results
from utils.helpers import pd_read_json
from config import Config

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

    STANDARD_WORD_SET = pd_read_json(Config.STANDARD_WORD_SET_PATH)
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

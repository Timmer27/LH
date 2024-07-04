import pandas as pd
import numpy as np
import json
import torch
import re
import os
from sentence_transformers import util
from sklearn.metrics import jaccard_score
from ckonlpy.tag import Twitter
from datetime import datetime

# 파일 불러오기
def file_load(file_name):
    print('file_loading..start')

    df = pd.read_csv(file_name, encoding='cp949')
    loaded_file = pd.DataFrame(columns=['contents'])
    loaded_file['contents'] = df[['하자내용']]
    
    print('file_loading..end')
    return loaded_file

# CSV 파일 로드
def pd_read_file(file_name):
    df = pd.read_csv(file_name, encoding='cp949')
    df = df.drop('Unnamed: 0', axis=1)
    df.reset_index(drop=True, inplace=True)
    return df

# Json 파일 로드
def pd_read_json(file_name):
    with open(file_name, encoding='utf-8-sig') as f:
        js = json.loads(f.read())
    return pd.DataFrame(js)

def preprocess_text(path, file_name, content):    
    # .txt 파일에서 변환할 단어들을 리스트로 불러옴
    with open(os.path.join(path, file_name), 'r', encoding='utf-8-sig') as f:
        replace_list = [line.strip() for line in f]

    # '침실' 다음에 오는 숫자들을 찾아서 각각 쉼표와 공백으로 구분
    content = re.sub(r'(침실)(\d+),(\d+),(\d+)', r'\1\2, \1\3, \1\4', content)
    content = re.sub(r'(침실)(\d+),(\d+)', r'\1\2, \1\3', content)

    # '침실' 다음에 오는 숫자가 아닌 숫자들을 제거
    content = re.sub(r'(?<=\D)(?<!침실)\d+', " ", content)

    # 리스트의 각 항목을 공백으로 바꿈
    for word in replace_list:
        content = re.sub(r'\b' + word + r'\b', " ", content)
    
    # 나머지 전처리 과정
    content = re.sub('\xa0', " ", content)
    content = re.sub("\n", " ", content)
    content = re.sub("\t", " ", content)
    content = re.sub("-", " ", content)
    content = re.sub("/", " ", content) # '/' 문자를 공백으로 바꿈
    content = re.sub("[\s]*\.[\s]+", " ", content)
    content = re.sub("[\.]{2,}", " ", content)
    content = re.sub("  ", " ", content)
    content = re.sub("  ", " ", content)
    content = re.sub("[^가-힣ㄱ-하-ㅣa-zA-Z침실\d\\s]", " ", content) # 한글, 영어, '침실' 다음에 오는 숫자 제외하고 모두 제거
    content = re.sub(r'\b\w\b', " ", content) # 1글자 단어 제거
    content = re.sub("\s+", " ", content)
    content = re.sub("  ", " ", content)
    content = re.sub("  ", " ", content)
    content = re.sub("  ", " ", content)
    
    # 양쪽 공백 제거
    content = content.strip()
    return content

# 표준화 처리 함수
def replace_words(text, idx, word_dict):
    # word_dict의 각 항목에 대해 반복
    for input_words, output_word in word_dict.items():
        # 입력 단어들을 쉼표로 분리하고 공백을 제거
        input_words = input_words.split(',')
        input_words = [word.strip() for word in input_words]
        # 각 입력 단어에 대해 반복
        for input_word in input_words:
            # 입력 단어를 출력 단어로 바꿈
            # '\b'는 단어 경계를 나타내는 정규 표현식 메타 문자
            if isinstance(input_word, str):
                # 단어가 이미 변경되었는지 확인
                if output_word not in text:
                    text = re.sub(r'\b' + input_word + r'\b', output_word, text)
    # 1000번째 마다 진행 상황을 출력
    if idx % 10000 == 0:
        print(f'Processed {idx} texts', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return text


# 일위대가 전처리 함수
def cost_preprocess_text(content):    
    # .txt 파일에서 변환할 단어들을 리스트로 불러옴
    with open('../data/dic/불용어.txt', 'r') as f:
        replace_list = [line.strip() for line in f]
    
    # 리스트의 각 항목을 공백으로 바꿈
    for word in replace_list:
        content = re.sub(r'\b' + word + r'\b', " ", content)
    
    # 나머지 전처리 과정
    content = re.sub('\xa0', " ", content)
    content = re.sub("\n", " ", content)
    content = re.sub("\t", " ", content)
    content = re.sub("-", " ", content)
    content = re.sub(r'\b' + '보수' + r'\b', ' ', content)
    content = re.sub(r'\b' + '침실1,2' + r'\b', '침실1, 침실2', content)
    content = re.sub("/", " ", content) # '/' 문자를 공백으로 바꿈
    content = re.sub("[\s]*\.[\s]+", " ", content)
    content = re.sub("[\.]{2,}", " ", content)
    content = re.sub("  ", " ", content)
    content = re.sub("  ", " ", content)
    content = re.sub("[^가-힣ㄱ-하-ㅣa-zA-Z\\s]", " ", content) # 한글 영어 제외하고 모두 제거
    content = re.sub(r'\b\w\b', " ", content) # 1글자 단어 제거
    content = re.sub("\s+", " ", content)
    content = re.sub("  ", " ", content)
    content = re.sub("  ", " ", content)
    content = re.sub("  ", " ", content)
    
    # 양쪽 공백 제거
    content = content.strip()
    return content

# # 필터 조건을 정의하는 함수
# def filter_data(df, embeddings, column_values, top_k):
#     # 조건을 순서대로 정의
#     conditions_order = [
#                         ['(신)공종', '(신)공간', '(신)부위자재', '(신)하자유형'],  # 첫 번째 조건: '(신)공종'과 '(신)공간'과 (신)부위자재와 (신)하자유형이 일치
#                         ['(신)공종', '(신)공간', '(신)부위자재'],  # 두 번째 조건: '(신)공종'과 '(신)공간'과 (신)부위자재가 일치
#                         ['(신)공종', '(신)공간'],  # 세 번째 조건: '(신)공종'과 '(신)공간'이 일치
#                         ['(신)공종']  # 네 번째 조건: '(신)공종'만 일치
#                         ]

#     # 각 조건에 대해 필터링 시도
#     for conditions in conditions_order:
#         condition_checks = [df[column] == column_values[column] for column in conditions]
#         filtered_df = df[np.logical_and.reduce(condition_checks)]
#         filtered_df = filtered_df.reset_index(drop=True)

#         # 필터링 결과가 비어있지 않으면 결과 반환
#         if len(filtered_df) >= top_k:
#             filtered_embeddings = embeddings[np.logical_and.reduce(condition_checks)]
#             return filtered_df, filtered_embeddings

#     # 모든 조건에 대해 필터링 결과가 비어있으면 원래의 데이터프레임과 임베딩 반환
#     return df, embeddings


# 필터 조건을 정의하는 함수
def filter_data(df, embeddings, column_values, top_k):
    # 조건을 순서대로 정의
    conditions_order = [
                        ['(신)공간', '(신)부위자재', '(신)하자유형'],  # 첫 번째 조건: '(신)공간'과 (신)부위자재와 (신)하자유형이 일치
                        ['(신)공간', '(신)부위자재'],  # 두 번째 조건: '(신)공간'과 (신)부위자재가 일치
                        ['(신)공간'],  # 세 번째 조건: '(신)공간'이 일치
                        ]

    # 각 조건에 대해 필터링 시도
    for conditions in conditions_order:
        condition_checks = [df[column] == column_values[column] for column in conditions]
        filtered_df = df[np.logical_and.reduce(condition_checks)]
        filtered_df = filtered_df.reset_index(drop=False)

        # 필터링 결과가 비어있지 않으면 결과 반환
        if len(filtered_df) >= top_k:
            filtered_indices = filtered_df['index'].values
            filtered_embeddings = embeddings[filtered_indices]
            filtered_df = filtered_df.drop(columns=['index'])
            return filtered_df, filtered_embeddings, filtered_indices

    # 모든 조건에 대해 필터링 결과가 비어있으면 원래의 데이터프레임과 임베딩 반환
    return df, embeddings, np.arange(len(df))


def initialize_morpheme_analyzer(standard_word_set):
    # 형태소 분석기 초기화
    twitter = Twitter()

    # 사용자 정의 사전 (리스트 형태)
    # 시리즈를 리스트로 변환하고 각 문자열을 쉼표로 분리
    standard_word_list = list(set(word.strip() for sublist in standard_word_set['INP_DTLS'].str.split(',') for word in sublist))
    
    # 사용자 정의 사전에 단어 추가
    for word in standard_word_list:
        twitter.add_dictionary(word, 'Noun')
    
    return twitter

def preprocess_corpus(corpus, twitter):
    preprocessed_corpus = []
    for i, text in enumerate(corpus):
        # 형태소 분석 수행
        morphs = twitter.morphs(text)
        preprocessed_corpus.append(morphs)

        # 진행 상황 출력 (100개 문서마다 한 번씩 출력)
        if (i+1) % 1000 == 0:
            print(f"Processed {i+1} documents.")
    return preprocessed_corpus

# 유사도 계산 함수 정의
def cosine_similarity(query_embedding, corpus_embeddings):
    return util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]

# def euclidean_similarity(query_embedding, corpus_embeddings):
#     return torch.sqrt(torch.sum((query_embedding - corpus_embeddings)**2, dim=1))  # 음수 제거

# def manhattan_similarity(query_embedding, corpus_embeddings):
#     return torch.sum(torch.abs(query_embedding - corpus_embeddings), dim=1)  # 음수 제거

def euclidean_similarity(query_embedding, corpus_embeddings):
    scores = torch.sqrt(torch.sum((query_embedding - corpus_embeddings)**2, dim=1))  # 음수 제거
    return scores / torch.max(scores)  # 정규화

def manhattan_similarity(query_embedding, corpus_embeddings):
    scores = torch.sum(torch.abs(query_embedding - corpus_embeddings), dim=1)  # 음수 제거
    return scores / torch.max(scores)  # 정규화


def jaccard_similarity(query, corpus):
    query = set(query)  # 형태소 분석 결과를 집합으로 변환
    scores = []
    for text in corpus:
        intersection = len(query.intersection(set(text)))
        union = len(query.union(set(text)))
        scores.append(intersection / union)
    return np.array(scores)

def overlap_similarity(query, corpus):
    query = set(query)  # 형태소 분석 결과를 집합으로 변환
    scores = []
    for text in corpus:
        intersection = len(query.intersection(set(text)))
        min_len = min(len(query), len(set(text)))
        if min_len == 0:  # query와 text가 모두 공집합인 경우
            scores.append(0)  # 유사도를 0으로 설정
        else:
            scores.append(intersection / min_len)
    return np.array(scores)


def cal_sim_results(query, query_embedding, corpus_embeddings, top_k, method, preprocessed_corpus):
    # 유사도 계산 방법에 따라 점수를 계산.
    if method in ['cosine', 'euclidean', 'manhattan']:                
        if method == 'cosine':
            scores = cosine_similarity(query_embedding, corpus_embeddings)
        elif method == 'euclidean':
            scores = euclidean_similarity(query_embedding, corpus_embeddings)
        elif method == 'manhattan':
            scores = manhattan_similarity(query_embedding, corpus_embeddings)
    elif method == 'jaccard':
        scores = jaccard_similarity(query, preprocessed_corpus)
        scores = torch.from_numpy(scores)  # jaccard_similarity는 numpy 배열을 반환하므로 텐서로 변환
    elif method == 'overlap':
        scores = overlap_similarity(query, preprocessed_corpus)
        scores = torch.from_numpy(scores)  # overlap_similarity는 numpy 배열을 반환하므로 텐서로 변환
    else:
        raise ValueError(f'Unknown method: {method}')

    # 점수를 디바이스에 맞게 변환하고, 상위 k개의 결과를 반환
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # GPU가 사용 가능한지 확인
    scores = scores.to(device)
    if method in ['euclidean', 'manhattan']:
        top_results = torch.topk(scores, k=top_k, largest=False).indices  # 'euclidean'과 'manhattan'에서는 점수가 낮은 결과가 더 유사
    else:
        top_results = torch.topk(scores, k=top_k).indices
    return top_results, scores[top_results]


def assign_workers(period):
    if pd.isnull(period):
        return ''
    if period <= 60:
        return 2
    elif period <= 120:
        return 3
    else:
        return 4


def calculate_duration(row):
    if pd.isna(row['k_보수완료보고일자']):
        if pd.isna(row['k_보수완료일자']):
            return row['k_보수예정일자'] - row['k_보수지시일']
        else:
            return row['k_보수완료일자'] - row['k_보수지시일']
    else:
        return row['k_보수완료보고일자'] - row['k_보수지시일']
    


def sort_and_remove_duplicates(df, columns_to_consider):
    # 각 칼럼의 값들이 각각 등장한 횟수에 따라 정렬
    for column in columns_to_consider:
        if column != '수량':
            df['count_' + column] = df.groupby(column)[column].transform('count')

    # 정렬 순서를 지정 (빈도수가 높은 순서, 그리고 칼럼의 값이 높은 순서)
    sort_order = []
    for column in columns_to_consider:
        if column != '수량':
            sort_order.extend(['count_' + column, column])
        else:
            sort_order.append(column)

    df = df.sort_values(by=sort_order, ascending=[False]*len(sort_order))

    # 칼럼들을 기준으로 중복 제거
    df = df.drop_duplicates(subset=columns_to_consider)
    df = df.drop(columns=['count_' + column for column in columns_to_consider if column != '수량'])

    return df

# API 응답 양식으로 변환
def convert_to_api_format(row):
    return {
        "unt_prc_cd": row['일위대가코드'],
        "unt_prc_nm": row['일위대가명'],
        "unt_prc_stdd": row['규격'],
        "unt_prc_unt": row['단위'],
        "unt_prc_qty": row['수량'],
        "unt_prc_mc": row['재료비'],
        "unt_prc_lc": row['노무비'],
        "unt_prc_e": row['경비'],
        "unt_prc_tc": row['합계']
    }
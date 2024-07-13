import torch, json
import pandas as pd

def tensor_to_serializable(data):
    if isinstance(data, torch.Tensor):
        return data.tolist()
    elif isinstance(data, (list, tuple)):
        return [tensor_to_serializable(item) for item in data]
    elif isinstance(data, dict):
        return {key: tensor_to_serializable(value) for key, value in data.items()}
    return data

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
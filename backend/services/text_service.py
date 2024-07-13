import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models.text_model import TextDataset
from utils.preprocessing import preprocess_text, replace_words

def txt_prd_func(model, flw_cts, word_dict, tokenizer, labels_df, flwDtlSn="01234567", flwDsCd="입주사사전방문(웰컴데이)", num=3):
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
    start_time = time.time()
    input_text = flw_cts
    pre_input_text = preprocess_text('stopwords.txt', input_text)
    stan_input_text = replace_words(pre_input_text, 1, word_dict)

    test_encodings = tokenizer([stan_input_text], truncation=True, padding=True)    
    test_dataset = TextDataset(test_encodings, [0])         
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    model.to(device)
    model.eval()        
    for batch in test_loader:  
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)

    probabilities = F.softmax(outputs.logits, dim=1)
    prob, indices = torch.topk(probabilities, num)

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

    key_mapping = {i: f"{_ordinal_suffix(i+1)}_tp" for i in range(num)}

    rspns = {
        "status": "success",
        "message": "Data processed successfully",
        "defect_info_detail": {
            "flwDtlSn": flwDtlSn,
            "flwDsCd": flwDsCd
        },
        "result": [
            {f"{key_mapping[i]}_tp_cd": item["tp_cd"],
             f"{key_mapping[i]}_tp_nm": item["tp_nm"],
             f"{key_mapping[i]}_tp_prb": item["tp_prb"]}
            for i, item in enumerate(result)
        ],
    }

    end_time = time.time()
    print("응답시간: ", end_time - start_time)
    return rspns

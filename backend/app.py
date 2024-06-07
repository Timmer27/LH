from flask import Flask, jsonify, make_response
# 내장 라이브러리
import os, sys, time, warnings
from datetime import datetime, timedelta

# 외장 라이브러리
import pandas as pd, numpy as np
import dask.dataframe as dd
from tqdm import tqdm
from ckonlpy.tag import Twitter

# 딥러닝 라이브러리
from sentence_transformers import SentenceTransformer, models, datasets, losses, util
from torch.utils.data import DataLoader
import torch
import json
import re
from sentence_transformers import util
from sklearn.metrics import jaccard_score
from ckonlpy.tag import Twitter
# 경고 무시
warnings.filterwarnings('ignore')

# 사용자 모듈
from common import *
from func_collection import *

app = Flask(__name__)

# 날짜 설정
today = '20240415'

# 경로 설정
src_path = '/home/tako/eoks/lh/lh_dev2/src'

# GPU 설정
# torch.cuda.set_device(1)

# 경로 추가
if src_path not in sys.path:
    sys.path.append(src_path)


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

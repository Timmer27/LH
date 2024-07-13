from flask import Blueprint
import torch, os
from config import Config
test_bp = Blueprint('test', __name__)

@test_bp.route('/gputest', methods=['GET'])
def gputest():
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
    model = torch.load(os.path.join(Config.SCHEDULE_MODEL_FILES, 'dev_BERT_TSDAE_MODEL_20240529.pt'), map_location=device)
    # 임베딩 벡터 로드
    train_embeddings = torch.load(os.path.join(Config.SCHEDULE_MODEL_FILES, 'dev_corpus_embedded_20240529.pt'),  map_location=device).float()

    return "hi test"

@test_bp.route('/test', methods=['GET'])
def test():
    device = torch.device('cuda:0')
    print("Using GPU:", torch.cuda.get_device_name(0))
    # 학습된 모델 로드
    model = torch.load(os.path.join(Config.SCHEDULE_MODEL_FILES, 'dev_BERT_TSDAE_MODEL_20240529.pt'))
    # 임베딩 벡터 로드
    train_embeddings = torch.load(os.path.join(Config.SCHEDULE_MODEL_FILES, 'dev_corpus_embedded_20240529.pt')).float()

    return torch.cuda.get_device_name(0)


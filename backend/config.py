import os

class Config:
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'source')
    OUTPUT_FOLDER = os.path.join(os.getcwd(), 'results')
    IMAGE_MODEL_FILES = os.path.join(os.getcwd(), 'static', 'image')
    TEXT_MODEL_FILES = os.path.join(os.getcwd(), 'static', 'text')
    SCHEDULE_MODEL_FILES = os.path.join(os.getcwd(), 'static', 'scheduling')
    AUTOMATED_MODEL_FILES = os.path.join(os.getcwd(), 'static', 'automation')
    STANDARD_WORD_SET_PATH = os.path.join(os.getcwd(), 'static', 'TB_MMA_CNV_DIC_M.json')
    STOPWORDS_PATH = os.path.join(os.getcwd(), 'static', 'stopwords.txt')
    
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

from flask import Blueprint, jsonify, request, send_from_directory
from config import Config
import os

main_bp = Blueprint('main', __name__)

@main_bp.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = file.filename
        file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(file_path)
        return jsonify({'file_name': filename, 'file_path': file_path, 'output_path': Config.OUTPUT_FOLDER}), 200

@main_bp.route('/image/<filename>')
def get_image(filename):
    image_directory = Config.OUTPUT_FOLDER
    return send_from_directory(image_directory, filename)

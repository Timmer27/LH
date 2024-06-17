FROM python:3.7.13-slim
# RUN mkdir /backend
WORKDIR /backend
COPY requirements.txt /backend/requirements.txt
# COPY common.py /backend/common.py
# COPY func_collection.py /backend/func_collection.py
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["flask", "run", "--host=0.0.0.0"]
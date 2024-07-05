FROM python:3.7.13-slim
WORKDIR /backend
COPY requirements.txt /backend/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .
# COPY common.py /backend/common.py
# COPY func_collection.py /backend/func_collection.py
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 openjdk-11-jdk && \
    apt-get clean

# Set JAVA_HOME environment variable
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64
ENV PATH $JAVA_HOME/bin:$PATH
    
COPY source /backend/source
COPY results /backend/results
EXPOSE 5000
CMD ["flask", "run", "--host=0.0.0.0"]
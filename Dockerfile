FROM python:3.11
COPY requirements.txt .
ENV PYTHONPATH=\/workspace
RUN apt-get update -y &&\
    apt-get -y install poppler-utils tesseract-ocr &&\
    git clone https://github.com/tesseract-ocr/tessdata.git /usr/share/tessdata &&\
    pip install -r requirements.txt --trusted-host pypi.python.org --no-cache-dir

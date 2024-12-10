FROM python:3.11
COPY requirements.txt .
ENV PYTHONPATH=\/workspace
ARG USER_ID
RUN apt-get update -y &&\
    apt-get -y install poppler-utils tesseract-ocr yq
RUN git clone https://github.com/tesseract-ocr/tessdata.git /usr/share/tessdata
RUN pip install -r requirements.txt --trusted-host pypi.python.org --no-cache-dir
RUN useradd -l -u ${USER_ID} -g sudo jenkins && \
    mkdir -m 0755 /home/jenkins && chown jenkins /home/jenkins
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download en_core_web_lg
USER jenkins
WORKDIR /workspace

FROM python:3.11
COPY requirements.txt .
ENV PYTHONPATH=\/workspace
ARG USER_ID
RUN apt-get update -y && \
    apt-get -y install poppler-utils tesseract-ocr yq &&\
    git clone https://github.com/tesseract-ocr/tessdata.git /usr/share/tessdata &&\
    useradd -l -u ${USER_ID} -g sudo jenkins && \
    mkdir -m 0755 /home/jenkins && chown jenkins /home/jenkins
USER jenkins
RUN pip install -r full.txt -r requirements.txt --trusted-host pypi.python.org --no-cache-dir &&\
    python -m spacy download en_core_web_sm && \
    python -m spacy download en_core_web_lg
ENV PATH="/home/jenkins/.local/bin:$PATH"
WORKDIR /workspace

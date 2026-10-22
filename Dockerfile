FROM python:3.11
COPY requirements.txt .
ENV PYTHONPATH=\/workspace
ARG USER_ID
RUN apt-get update -y && \
    apt-get -y install poppler-utils tesseract-ocr yq build-essential libopenblas-dev libomp-dev &&\
    git clone --depth 1 https://github.com/tesseract-ocr/tessdata.git /usr/share/tessdata &&\
    useradd -l -u ${USER_ID} -g sudo jenkins && \
    mkdir -m 0755 /home/jenkins && chown jenkins /home/jenkins
USER jenkins
# The import check fails the build if the PII scrubber's dependencies are ever dropped from requirements.txt.
RUN pip install -r requirements.txt --trusted-host pypi.python.org --no-cache-dir &&\
    python -c "import torch, transformers, spacy_huggingface_pipelines" && \
    python -m spacy download en_core_web_sm && \
    python -m spacy download en_core_web_lg
ENV PATH="/home/jenkins/.local/bin:$PATH"
WORKDIR /workspace

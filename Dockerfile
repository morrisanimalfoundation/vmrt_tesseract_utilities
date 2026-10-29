FROM python:3.11
# Links the published image on ghcr.io to this repository.
LABEL org.opencontainers.image.source=https://github.com/morrisanimalfoundation/vmrt_tesseract_utilities
ENV PYTHONPATH=\/workspace
# Keeps pip's download cache (including the spaCy model downloads) out of the image.
ENV PIP_NO_CACHE_DIR=1
# Only the English and orientation models are needed: the OCR code uses Tesseract's default language.
# Cloning the whole tesseract-ocr/tessdata repository would add about 2 GB of other languages.
ARG TESSDATA_COMMIT=ced78752cc61322fb554c280d13360b35b8684e4
RUN apt-get update -y && \
    apt-get -y install poppler-utils tesseract-ocr yq build-essential libopenblas-dev libomp-dev && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /usr/share/tessdata && \
    for model in eng osd; do \
      curl -fsSL --retry 3 -o /usr/share/tessdata/$model.traineddata \
        https://github.com/tesseract-ocr/tessdata/raw/${TESSDATA_COMMIT}/$model.traineddata || exit 1; \
    done && \
    printf '%s  %s\n' \
      daa0c97d651c19fba3b25e81317cd697e9908c8208090c94c3905381c23fc047 /usr/share/tessdata/eng.traineddata \
      e19f2ae860792fdf372cf48d8ce70ae5da3c4052962fe22e9de1f680c374bb0e /usr/share/tessdata/osd.traineddata \
      | sha256sum -c -
COPY requirements.txt .
# The import check fails the build if the PII scrubber's dependencies are ever dropped from requirements.txt.
RUN pip install -r requirements.txt &&\
    python -c "import torch, transformers, spacy_huggingface_pipelines" && \
    python -m spacy download en_core_web_sm && \
    python -m spacy download en_core_web_lg
# The container runs as the host user (see docker-compose.yml), which has no account in the image, so give
# it a writable home for caches such as the Hugging Face model download, a name for tools that ask, and a
# prompt that doesn't say "I have no name!".
RUN mkdir -m 1777 /home/vmrt && \
    echo 'PS1="vmrt@\h:\w\$ "' >> /etc/bash.bashrc
ENV HOME=/home/vmrt USER=vmrt
WORKDIR /workspace

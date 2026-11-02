# The slim image has no compilers or C headers. Every package in requirements.txt has a prebuilt wheel for
# both linux/amd64 and linux/arm64, so none are needed (pip install below refuses to build one from source), and
# leaving them out keeps their vulnerabilities (such as kernel header CVEs in linux-libc-dev) out of the image.
FROM python:3.11-slim
# Links the published image on ghcr.io to this repository.
LABEL org.opencontainers.image.source=https://github.com/morrisanimalfoundation/vmrt_tesseract_utilities
ENV PYTHONPATH=\/workspace
# Keeps pip's download cache (including the spaCy model downloads) out of the image.
ENV PIP_NO_CACHE_DIR=1
# Only the English and orientation models are needed: the OCR code uses Tesseract's default language.
# Cloning the whole tesseract-ocr/tessdata repository would add about 2 GB of other languages.
ARG TESSDATA_COMMIT=ced78752cc61322fb554c280d13360b35b8684e4
# The upgrade installs Debian security fixes released since the python:3.11-slim base image was last rebuilt. The
# options keep the build from stopping to ask about a changed config file, which would fail with no one to answer.
# poppler-utils is used by pdf2image, and poppler-data lets it render PDFs whose fonts use predefined character
# maps. Tesseract itself comes with the tesserocr wheel, so it isn't installed here. procps and less are for the
# interactive shell (ps, less). curl is only needed to download the Tesseract models, so it's removed again in the
# same step.
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get -y -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold upgrade && \
    apt-get -y install --no-install-recommends poppler-utils poppler-data procps less curl && \
    mkdir -p /usr/share/tessdata && \
    for model in eng osd; do \
      curl -fsSL --retry 3 -o /usr/share/tessdata/$model.traineddata \
        https://github.com/tesseract-ocr/tessdata/raw/${TESSDATA_COMMIT}/$model.traineddata || exit 1; \
    done && \
    printf '%s  %s\n' \
      daa0c97d651c19fba3b25e81317cd697e9908c8208090c94c3905381c23fc047 /usr/share/tessdata/eng.traineddata \
      e19f2ae860792fdf372cf48d8ce70ae5da3c4052962fe22e9de1f680c374bb0e /usr/share/tessdata/osd.traineddata \
      | sha256sum -c - && \
    apt-get -y purge --auto-remove curl && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# The import check fails the build if the PII scrubber's dependencies are ever dropped from requirements.txt.
RUN pip install --only-binary=:all: -r requirements.txt && \
    python -c "import torch, transformers, presidio_analyzer.nlp_engine.huggingface_token_pipe" && \
    python -m spacy download en_core_web_sm && \
    python -m spacy download en_core_web_lg
# The container runs as the host user (see docker-compose.yml), which has no account in the image, so give
# it a writable home for caches such as the Hugging Face model download, a name for tools that ask, and a
# prompt that doesn't say "I have no name!".
RUN mkdir -m 1777 /home/vmrt && \
    echo 'PS1="vmrt@\h:\w\$ "' >> /etc/bash.bashrc
ENV HOME=/home/vmrt USER=vmrt
WORKDIR /workspace
# Installs Debian security fixes released since the layers above were cached. CI passes the date as APT_REFRESH,
# so this layer, and only this one, is rebuilt once a day, and a fix for a newly published vulnerability reaches
# pull request builds the next day rather than at the weekly rebuild. Local builds don't set it.
ARG APT_REFRESH=
RUN echo "Refreshing Debian packages (${APT_REFRESH:-no date given})" && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get -y -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold upgrade && \
    rm -rf /var/lib/apt/lists/*

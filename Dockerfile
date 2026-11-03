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
# The PII scrubber's model, pinned to one revision so its scrubbing only changes when this pin does. Presidio's NLP
# config can't set a revision, so the model is downloaded here into a fixed folder, which
# scripts/scrubbers/config/stanford-deidentifier-base_nlp.yaml points at, and the weights are checked against their
# SHA-256 (from Hugging Face's file page for that revision). HF_HUB_OFFLINE stops anything from being fetched from
# Hugging Face at run time, so the scrubber works without internet and fails rather than downloading a different
# model. To update the pin, see "Updating the PII scrubber's model" in the README.
ARG SCRUBBER_MODEL=StanfordAIMI/stanford-deidentifier-base
ARG SCRUBBER_MODEL_REVISION=661b9c1c717d3165512d440abc3700c386aefab6
ARG SCRUBBER_MODEL_SHA256=fa49ef069171e479f546ce2ee5ed599aa585d1d33bc7a8f54400ac57d9cd2716
ARG SCRUBBER_MODEL_DIR=/opt/models/stanford-deidentifier-base
RUN python -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])" \
      "$SCRUBBER_MODEL" "$SCRUBBER_MODEL_REVISION" "$SCRUBBER_MODEL_DIR" && \
    printf '%s  %s\n' "$SCRUBBER_MODEL_SHA256" "$SCRUBBER_MODEL_DIR/pytorch_model.bin" | sha256sum -c - && \
    rm -rf "$SCRUBBER_MODEL_DIR/.cache" && \
    echo "$SCRUBBER_MODEL@$SCRUBBER_MODEL_REVISION" > "$SCRUBBER_MODEL_DIR/REVISION"
ENV HF_HUB_OFFLINE=1
# The container runs as the host user (see docker-compose.yml), which has no account in the image, so give
# it a writable home for caches and `pip install --user`, a name for tools that ask, and a
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

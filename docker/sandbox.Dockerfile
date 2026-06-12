FROM python:3.13-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        lsof \
        nodejs \
        npm \
        procps \
        tmux \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
        Pillow>=9.0 \
        py-vapid>=1.9.2 \
        pywebpush>=2.3.0 \
        pytest

RUN useradd -ms /bin/bash tester

WORKDIR /workspace
USER tester

ENV HOME=/home/tester \
    PYTHONPATH=/workspace

CMD ["python3", "-m", "codoxear.server"]

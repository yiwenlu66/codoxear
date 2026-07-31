FROM node:25-slim AS node_runtime

FROM python:3.13-slim

ARG HOST_UID=1000
ARG HOST_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/tester \
    PYTHONPATH=/workspace \
    PATH=/home/tester/.npm-global/bin:/opt/codoxear-host-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    CODEX_WEB_PORT=19580

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        ripgrep \
        tmux \
        zsh \
        zoxide \
    && rm -rf /var/lib/apt/lists/*

COPY --from=node_runtime /usr/local/bin/node /usr/local/bin/node
COPY --from=node_runtime /usr/local/bin/npm /usr/local/bin/npm
COPY --from=node_runtime /usr/local/bin/npx /usr/local/bin/npx
COPY --from=node_runtime /usr/local/lib/node_modules /usr/local/lib/node_modules

# Docker COPY resolves the upstream npm/npx symlinks, so recreate them against
# the copied module tree rather than leaving launcher scripts in /usr/local/bin.
RUN ln -sf ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -sf ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

RUN python3 -m pip install --no-cache-dir \
        'Pillow>=9.0' \
        'py-vapid>=1.9.2' \
        'pywebpush>=2.3.0'

RUN groupadd --gid "${HOST_GID}" tester \
    && useradd --uid "${HOST_UID}" --gid "${HOST_GID}" --create-home --shell /usr/bin/zsh tester \
    && mkdir -p /opt/codoxear-host-bin /run/host-config \
    && chown tester:tester /opt/codoxear-host-bin

WORKDIR /workspace
USER tester

HEALTHCHECK --interval=5s --timeout=3s --start-period=10s --retries=12 \
  CMD test "$(curl -sS -o /dev/null -w '%{http_code}' "http://127.0.0.1:${CODEX_WEB_PORT}/api/me")" = 401

CMD ["python3", "-m", "codoxear.server"]

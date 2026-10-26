# Isolated behavioural verifier image for the sidebar → tmux order mirror.
#
# Self-contained on purpose: unlike the other verify images this one needs no
# agent CLI. The container runs the real codoxear server plus real tmux and
# launches sessions whose backend binary is a stub (`PI_BIN`), which is enough
# for the window bookkeeping the mirror acts on.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends tmux ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
        'Pillow>=9.0' 'py-vapid>=1.9.2' 'pywebpush>=2.3.0' 'websockets>=14,<17'

COPY . /workspace
COPY docker/tmux_order_driver.py /opt/tmux_order_driver.py

RUN useradd --create-home --uid 1000 tester \
    && chown -R tester:tester /home/tester \
    && chmod -R a+rX /workspace /opt/tmux_order_driver.py

USER tester
ENV HOME=/home/tester \
    PYTHONPATH=/workspace \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace

CMD ["python3", "/opt/tmux_order_driver.py"]

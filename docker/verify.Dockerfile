# Real Pi is installed separately so verification can remain based on the
# pre-existing isolated application image while running a fresh Pi session.
FROM node:25-slim AS pi_runtime

RUN npm install --global @earendil-works/pi-coding-agent@0.82.1

FROM codoxear-iso3:latest

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libatomic1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=pi_runtime /usr/local/bin/node /usr/local/bin/node
COPY --from=pi_runtime /usr/local/bin/npm /usr/local/bin/npm
COPY --from=pi_runtime /usr/local/bin/npx /usr/local/bin/npx
COPY --from=pi_runtime /usr/local/lib/node_modules /usr/local/lib/node_modules

# Docker COPY resolves Node's npm/npx symlinks. Recreate them against the
# copied global package tree, including Pi's executable symlink.
RUN ln -sf ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -sf ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && ln -sf ../lib/node_modules/@earendil-works/pi-coding-agent/dist/cli.js /usr/local/bin/pi

COPY . /workspace
RUN chmod -R a+rX /workspace

WORKDIR /workspace
USER tester

ENV HOME=/home/tester \
    PYTHONPATH=/workspace \
    PYTHONDONTWRITEBYTECODE=1

CMD ["python3", "-m", "codoxear.server"]

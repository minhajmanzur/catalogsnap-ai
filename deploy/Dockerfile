# One image for both roles; the command picks API or worker.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    U2NET_HOME=/models

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Bake the cutout model into the image so a new pod can take work immediately
# instead of downloading weights on its first job.
ARG CUTOUT_MODEL=birefnet-general-lite
ENV CATALOGSNAP_CUTOUT_MODEL=${CUTOUT_MODEL}
RUN python -c "import os; from catalogsnap.cutout import _session; _session(os.environ['CATALOGSNAP_CUTOUT_MODEL'])"

RUN useradd --create-home app && chown -R app /app /models
USER app

EXPOSE 8000
CMD ["catalogsnap", "serve", "--host", "0.0.0.0", "--port", "8000"]

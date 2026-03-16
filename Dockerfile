FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    fontconfig \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libmagic1 \
    libsm6 \
    libxi6 \
    libxinerama1 \
    libxext6 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    libxrender1 \
    libxxf86vm1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt /tmp/requirements-docker.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install torchvision==0.17.2 && \
    pip install -r /tmp/requirements-docker.txt

COPY . /workspace

EXPOSE 8888

CMD ["sh", "-c", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --IdentityProvider.token=${JUPYTER_TOKEN:-devtoken} --PasswordIdentityProvider.hashed_password="]

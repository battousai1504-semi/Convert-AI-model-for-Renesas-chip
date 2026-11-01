FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ca-certificates libgl1-mesa-glx libglib2.0-0 libgomp1 make \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install numpy==1.26.4 Pillow==10.4.0 && \
    python3 -m pip install torch==2.3.1+cpu torchvision==0.18.1+cpu \
      --index-url https://download.pytorch.org/whl/cpu && \
    python3 -m pip install dx-com==2.4.1 torch==2.3.1+cpu torchvision==0.18.1+cpu \
      numpy==1.26.4 Pillow==10.4.0 onnx==1.16.1 onnxruntime==1.18.1 opencv-python==4.10.0.84 && \
    python3 -m pip check && dxcom --version
WORKDIR /work

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    VGGT_ROOT=/opt/vggt

RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates python3 python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install \
      torch==2.3.1 torchvision==0.18.1 \
      --index-url https://download.pytorch.org/whl/cu121 \
    && python3 -m pip install \
      numpy==1.26.1 Pillow huggingface_hub einops safetensors

RUN git clone https://github.com/facebookresearch/vggt.git ${VGGT_ROOT} \
    && git -C ${VGGT_ROOT} checkout a288dd0f14786c93483e45524328726ab7b1b4ce \
    && test "$(git -C ${VGGT_ROOT} rev-parse HEAD)" = "a288dd0f14786c93483e45524328726ab7b1b4ce"

WORKDIR /workspace/reference-worlds
COPY . /workspace/reference-worlds
RUN python3 -m pip install -e '.[dev,method]'

# VGGT weights (facebook/VGGT-1B) are intentionally downloaded at runtime so
# the container build does not silently freeze or redistribute model weights.
# Mount a Hugging Face cache for repeat runs if desired.

CMD ["bash"]

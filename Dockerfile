# Inference image for the thermal plate surrogate.
#   docker build -t plate-surrogate .
#   docker run --rm plate-surrogate info
#   docker run --rm plate-surrogate predict --k 2 --amp 3000 --x0 0.4 --y0 0.6 --sigma 0.1 \
#       --t-left 20 --t-right 80 --t-bottom 40 --t-top 10
#   docker run --rm plate-surrogate --model unet predict ...   (U-Net ensemble + conformal gate)
FROM python:3.12-slim

# Keep logs unbuffered and pip quiet; one thread pool is plenty for inference.
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

WORKDIR /app

# 1. CPU-only PyTorch first (~200 MB instead of the ~2 GB CUDA build). This layer
#    is cached, so code changes don't re-download it.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. The package itself.
COPY pyproject.toml ./
COPY src ./src
RUN pip install .

# 3. The trained models: POD + MLP (default) and the U-Net ensemble with its conformal gate.
COPY model ./model
COPY model_unet ./model_unet
ENV PLATE_MODEL_DIR=/app/model PLATE_UNET_DIR=/app/model_unet

# COPY keeps the host's file permissions; make sure the non-root user can read the models.
RUN chmod -R a+rX /app/model /app/model_unet

# Run as a non-root user.
RUN useradd --create-home app
USER app

ENTRYPOINT ["plate-surrogate"]
CMD ["--help"]

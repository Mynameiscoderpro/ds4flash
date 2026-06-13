# ---- builder ----
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake build-essential libcurl4-openssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN git clone --depth 1 https://github.com/Fringe210/llama.cpp-deepseek-v4-flash-cuda.git llama.cpp
WORKDIR /app/llama.cpp
# GitHub's builder has no NVIDIA GPU, so the driver file libcuda.so.1 is missing.
# Point the linker at the CUDA "stub" so llama-server can finish linking.
# The real driver is used at runtime on HuggingFace's GPUs.
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
ENV LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}"
RUN cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build --config Release -j"$(nproc)" --target llama-server

# ---- runtime ----
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      libcurl4 libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=build /app/llama.cpp/build/bin/ /opt/llama/
ENV LD_LIBRARY_PATH=/opt/llama
EXPOSE 80
ENTRYPOINT ["/opt/llama/llama-server", \
  "-m", "/repository/Huihui-DeepSeek-V4-Flash-BF16-abliterated-ds4-Q4_K.gguf", \
  "--host", "0.0.0.0", "--port", "80", "-c", "32768", \
  "--n-gpu-layers", "999", "--flash-attn", "on", \
  "--parallel", "1", "--no-warmup"]

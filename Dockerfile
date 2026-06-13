# ---- builder ----
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake build-essential libcurl4-openssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN git clone --depth 1 https://github.com/Fringe210/llama.cpp-deepseek-v4-flash-cuda.git llama.cpp
WORKDIR /app/llama.cpp
# GitHub's builder has no NVIDIA GPU, so the driver library libcuda.so.1 is missing.
# Make a stand-in from CUDA's bundled stub, register that folder with the system
# linker, and point the linker straight at it. The real driver is used later,
# at runtime, on HuggingFace's GPUs.
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
 && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf \
 && ldconfig
RUN cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
 && cmake --build build --config Release -j"$(nproc)" --target llama-server

# ---- runtime ----
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      libcurl4 libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=build /app/llama.cpp/build/bin/ /opt/llama/
ENV LD_LIBRARY_PATH=/opt/llama:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
EXPOSE 80
ENTRYPOINT ["/opt/llama/llama-server", \
  "-m", "/repository/Huihui-DeepSeek-V4-Flash-BF16-abliterated-ds4-Q4_K.gguf", \
  "--host", "0.0.0.0", "--port", "80", "-c", "32768", \
  "--n-gpu-layers", "999", "--flash-attn", "on", \
  "--parallel", "1", "--no-warmup"]

# F5-TTS Windows 部署文档

## 项目概述

F5-TTS 是一个基于 Flow Matching 的高质量语音合成系统，支持多种部署方式和 API 接口。本文档详细介绍在 Windows 环境下的完整部署流程。

## 系统要求

### 硬件要求
- **GPU**: 推荐 NVIDIA GPU（支持 CUDA 12.4+）
- **内存**: 至少 8GB RAM
- **存储**: 至少 10GB 可用空间

### 软件要求
- **操作系统**: Windows 10/11
- **Python**: 3.10
- **CUDA**: 12.4+ (如使用 GPU)

## 环境准备

### 1. 创建 Python 环境

```powershell
# 使用 conda 创建虚拟环境
conda create -n f5-tts python=3.10
conda activate f5-tts
```

### 2. 安装 PyTorch

```powershell
# NVIDIA GPU 用户
pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

# CPU 用户
pip install torch torchaudio
```

### 3. 安装 F5-TTS

#### 方式一：pip 安装（推荐用于推理）
```powershell
pip install f5-tts
```

#### 方式二：源码安装（推荐用于开发）
```powershell
git clone https://github.com/SWivid/F5-TTS.git
cd F5-TTS
pip install -e .
```

## API 启动方式

### 1. Gradio Web 界面（推荐）

#### 基本启动
```powershell
# 使用默认配置启动
f5-tts_infer-gradio

# 指定端口和主机
f5-tts_infer-gradio --port 7860 --host 0.0.0.0

# 启用公共分享链接
f5-tts_infer-gradio --share

# 启用 API 访问
f5-tts_infer-gradio --api
```

#### 高级配置
```powershell
# 完整配置示例
f5-tts_infer-gradio --port 7860 --host 0.0.0.0 --share --api --inbrowser
```

**访问地址**: `http://localhost:7860`

#### 支持功能
- 基础 TTS 合成
- 多风格/多说话人生成
- 语音聊天（基于 Qwen2.5-3B-Instruct）
- 自定义推理配置

### 2. Socket 服务器（流式推理）

#### 启动命令
```powershell
# 基本启动
python src/f5_tts/socket_server.py

# 自定义配置
python src/f5_tts/socket_server.py --host 0.0.0.0 --port 9998 --model F5TTS_v1_Base
```

#### 完整参数配置
```powershell
python src/f5_tts/socket_server.py \
    --host 0.0.0.0 \
    --port 9998 \
    --model F5TTS_v1_Base \
    --ref_audio "path/to/reference.wav" \
    --ref_text "参考音频的文本内容" \
    --device cuda
```

#### 客户端连接示例
```python
import asyncio
from f5_tts.socket_client import listen_to_F5TTS

# 异步调用
asyncio.run(listen_to_F5TTS(
    text="要合成的文本",
    server_ip="localhost",
    server_port=9998
))
```

### 3. CLI 命令行接口

#### 基本使用
```powershell
# 使用默认配置
f5-tts_infer-cli

# 指定参数
f5-tts_infer-cli --model F5TTS_v1_Base \
    --ref_audio "reference.wav" \
    --ref_text "参考音频的转录文本" \
    --gen_text "要生成的语音文本"
```

#### 使用配置文件
```powershell
# 使用自定义配置文件
f5-tts_infer-cli -c custom.toml

# 多语音生成
f5-tts_infer-cli -c src/f5_tts/infer/examples/multi/story.toml
```

### 4. Python API 调用

#### 基本使用示例
```python
from f5_tts.api import F5TTS

# 初始化模型
f5tts = F5TTS(
    model="F5TTS_v1_Base",
    device="cuda"  # 或 "cpu"
)

# 语音合成
wav, sr, spec = f5tts.infer(
    ref_file="reference.wav",
    ref_text="参考音频的文本内容",
    gen_text="要合成的目标文本",
    file_wave="output.wav",
    file_spec="output.png",
    seed=42
)

print(f"生成完成，采样率: {sr}")
print(f"随机种子: {f5tts.seed}")
```

### 5. Triton + TensorRT-LLM 部署（高性能）

#### 环境准备
```powershell
# 安装额外依赖
pip install tritonclient[grpc] tensorrt-llm==0.16.0
```

#### 启动 Triton 服务器
```bash
# 在 WSL 或 Linux 容器中运行
cd src/f5_tts/runtime/triton_trtllm
bash run.sh
```

#### HTTP 客户端调用
```powershell
python src/f5_tts/runtime/triton_trtllm/client_http.py \
    --server-url localhost:8000 \
    --reference-audio "reference.wav" \
    --reference-text "参考文本" \
    --target-text "目标文本" \
    --output-audio "output.wav"
```

#### gRPC 客户端调用
```powershell
python src/f5_tts/runtime/triton_trtllm/client_grpc.py \
    --server-addr localhost \
    --server-port 8001 \
    --reference-audio "reference.wav" \
    --reference-text "参考文本" \
    --target-text "目标文本"
```

## Docker 部署（可选）

### 1. 构建镜像
```powershell
docker build -t f5tts:v1 .
```

### 2. 运行容器
```powershell
# 运行 Gradio 界面
docker run --rm -it --gpus=all \
    -v f5-tts:/root/.cache/huggingface/hub/ \
    -p 7860:7860 \
    f5tts:v1 f5-tts_infer-gradio --host 0.0.0.0
```

### 3. Docker Compose 配置
```yaml
services:
  f5-tts:
    image: ghcr.io/swivid/f5-tts:main
    ports:
      - "7860:7860"
    environment:
      GRADIO_SERVER_PORT: 7860
    entrypoint: ["f5-tts_infer-gradio", "--port", "7860", "--host", "0.0.0.0"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 性能优化建议

### 1. GPU 加速
- 确保安装正确的 CUDA 版本
- 使用 `--device cuda` 参数启用 GPU
- 监控 GPU 内存使用情况

### 2. 内存优化
- 对于大批量处理，考虑使用批处理模式
- 及时释放不需要的模型资源
- 使用适当的数据类型（如 float16）

### 3. 网络配置
- 生产环境建议使用反向代理（如 Nginx）
- 配置适当的超时时间
- 启用 HTTPS（生产环境）

## 常见问题排查

### 1. 模型下载问题
```powershell
# 手动下载模型
huggingface-cli download SWivid/F5-TTS F5TTS_v1_Base/model_1250000.safetensors
```

### 2. CUDA 相关错误
```powershell
# 检查 CUDA 安装
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"
```

### 3. 端口占用问题
```powershell
# 检查端口占用
netstat -ano | findstr :7860

# 终止占用进程
taskkill /PID <PID> /F
```

### 4. 内存不足
- 减少批处理大小
- 使用 CPU 模式
- 关闭其他占用内存的程序

## 监控和日志

### 1. 启用详细日志
```python
import logging
logging.basicConfig(level=logging.INFO)
```

### 2. 性能监控
- 使用 `nvidia-smi` 监控 GPU 使用率
- 监控内存使用情况
- 记录推理时间和质量指标

## 安全建议

1. **网络安全**
   - 不要在公网直接暴露服务
   - 使用防火墙限制访问
   - 配置适当的认证机制

2. **资源限制**
   - 设置合理的并发限制
   - 限制单次请求的文本长度
   - 监控系统资源使用

3. **数据安全**
   - 不要记录敏感音频内容
   - 定期清理临时文件
   - 加密存储重要数据

## 总结

本文档提供了 F5-TTS 在 Windows 环境下的完整部署方案，包括：

- ✅ **Gradio Web 界面**: 适合交互式使用和演示
- ✅ **Socket 服务器**: 适合流式推理和实时应用
- ✅ **CLI 工具**: 适合批处理和脚本化调用
- ✅ **Python API**: 适合集成到其他应用中
- ✅ **Triton 部署**: 适合高性能生产环境

根据具体需求选择合适的部署方式，并按照文档进行配置和优化。如遇问题，请参考常见问题排查部分或查看项目的 GitHub Issues。
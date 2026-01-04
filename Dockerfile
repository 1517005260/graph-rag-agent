# 使用 x86 架构的 Python 镜像（避免 ARM 兼容性问题）
FROM --platform=linux/amd64 python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖（使用 x86 版本，避免段错误）
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 安装项目
RUN pip install -e .

# 创建必要的目录
RUN mkdir -p files/ cache/ frontend/ server/

# 暴露端口
EXPOSE 8000 8501

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 默认命令（可以被 docker-compose 覆盖）
CMD ["python", "server/main.py"]

# Backend Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 複製 requirements.txt（從 environment_info.txt 生成）
COPY requirements.txt .

# 安裝 Python 依賴（先安裝 PyTorch）
RUN pip install --no-cache-dir torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 然後安裝其他依賴
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式代碼
COPY . .

# 創建上傳和結果目錄
RUN mkdir -p uploads results static

# 暴露端口
EXPOSE 5001

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5001/api/health || exit 1

# 啟動命令
CMD ["python", "app.py"]
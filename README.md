# ai_defense_system

<img width="1904" height="906" alt="image" src="https://github.com/user-attachments/assets/9419a0f9-60d9-45df-a56d-0a63a130654c" />


## 專案目錄結構

```
security-defense-system/
├── backend/                     # Flask 後端
│   ├── app.py
│   ├── llama_jailbreak_detector/   # LLM model         
│   ├── enhanced_defense_system.py
│   ├── enhanced_llm_moderator.py
│   ├── requirements.txt     
│   ├── uploads/             
│   ├── static/               
│   └── venv/                 
│
├── frontend/                   
│   ├── public/
│   │   ├── index.html
│   │   └── favicon.ico
│   ├── src/
│   │   ├── components/       
│   │   │   ├── Dashboard.js
│   │   │   ├── RealTimeTest.js
│   │   │   ├── TestHistory.js
│   │   │   ├── BatchAnalysis.js
│   │   │   └── Settings.js
│   │   │   └── MultiModelTest.js
│   │   ├── App.js            
│   │   ├── App.css         
│   │   └── index.js         
│   ├── package.json          
│   └── .env                
├── models/                   
│   ├── toxigen_model/
└── README.md                
```

## 第一步：建立專案目錄

### 1.1 git clone project

```bash
cd ai_defense_system
```
### 1.2 在backend/app.py修改model路徑

```bash
system_config = {
    "model_path": "C:/Users/user/Desktop/rnn/toxigen_model",
    "ollama_url": "http://localhost:11434",
    "ollama_model": "llama3.1:8b",
    "toxicity_threshold": 0.4,
    "harmfulness_threshold": 0.3,
    "replacement_threshold": "low",
    "enable_ollama_replacement": True,
    "strict_mode": False
}
```
#### 請把model_path改成toxigen_model放置的路徑

toxigen_model downoload link: [LINK](https://drive.google.com/drive/folders/14uaWo6KFr4Agwti33J-8RLykP61mJOTw?usp=sharing)

#### 請下載llama_jailbreak_detector，並將整個資料夾和app.py放在同一層級

llama_jailbreak_detector downoload link: [LINK](https://drive.google.com/drive/folders/19dCxaDrI_2tSibpmEb-_EqLo8zKQ9KYs?usp=sharing)

## 第二步：設置後端 (Flask)

### 2.1 創建虛擬環境

```bash
cd backend

# 創建 Python 虛擬環境
python -m venv venv

# 啟動虛擬環境
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate
```

### 2.2 安裝 Python 依賴

```bash
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

```bash
pip install -r requirements_local.txt
```

- transformers: 4.52.3
- tokenizers: 0.21.1
- torch: 2.7.0+cu118 (GPU版本)


## 第三步：設置前端 (React)

### 3.1 創建 React 應用

```bash
# 回到專案根目錄
cd ..

# 創建 React 應用
npx create-react-app frontend
cd frontend
```

### 3.2 安裝 React 依賴

```bash
npm install antd axios recharts moment @ant-design/icons react-router-dom
```


## 第四步：啟動系統

### 4.1 啟動後端服務器

```bash
# 在 backend/ 目錄
cd backend

# 啟動虛擬環境
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# 本地登入 HF：Access to model meta-llama/Llama-3.2-3B-Instruct(需要有這個權限，要request Llama-3.2-3B-Instruct 的 access，token authority可以使用read就好)
huggingface-cli login
接著輸入自己的hf token

# 運行 Flask 服務器
python app.py
```

後端將運行在 http://localhost:5001

### 4.2 啟動前端開發服務器

開啟新的終端視窗：

```bash
# 在 frontend/ 目錄
cd frontend

# 安裝依賴 (如果還沒安裝)
npm install

# 啟動 React 開發服務器
npm start
```

前端將運行在 http://localhost:3000

## 第五步：配置 Ollama (可選)

如果要使用 Ollama 功能：

```bash
# 安裝 Ollama
# 從 https://ollama.ai 下載並安裝

# 啟動 Ollama 服務
ollama serve

# 下載推薦模型
ollama pull llama3.1:8b
ollama pull llama2:7b-chat
ollama pull mistral:7b-instruct
```

## 第六步：系統測試

1. 打開瀏覽器訪問 http://localhost:3000
2. 檢查系統狀態頁面
3. 測試實時檢測功能
4. 上傳 CSV 文件測試批量分析

## 開發工作流程

### 正常開發流程

1. **同時運行兩個服務器**:
   - Terminal 1: `cd backend && venv\Scripts\activate && python app.py`
   - Terminal 2: `cd frontend && npm start`

2. **API 測試**:
   - 可以直接訪問 http://localhost:5001/api/status 測試 API

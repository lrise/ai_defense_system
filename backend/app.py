from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
import io
import base64
from typing import Dict, List, Optional
import requests
import threading
import time
import numpy as np

from enhanced_llm_moderator import EnhancedContentModeratorV2


# 導入你的防禦系統
try:
    from enhanced_defense_system import (
        EnhancedContentModerator, 
        OllamaClient,
        evaluate_defense_system_with_ollama
    )
except ImportError as e:
    print(f"Warning: Could not import defense system: {e}")
    EnhancedContentModerator = None

app = Flask(__name__)
CORS(app)  # 允許跨域請求

# 配置
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# 確保目錄存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# 全局變量 - 在函數外部聲明
moderator = None
llm_moderator = None
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

# 數據存儲
test_history = []
batch_analysis_cache = {}

def initialize_defense_system():
    """初始化防禦系統 - 使用 LLM 增強版"""
    global moderator, llm_moderator
    
    if EnhancedContentModerator is None:
        return False
    
    try:
        config = {
            "toxicity_threshold": system_config["toxicity_threshold"],
            "harmfulness_threshold": system_config["harmfulness_threshold"],
            "enable_ollama_replacement": system_config["enable_ollama_replacement"],
            "replacement_threshold": system_config["replacement_threshold"],
            "strict_mode": system_config["strict_mode"],
            "enable_sanitization": True,
            "ollama_url": system_config["ollama_url"],
            "ollama_model": system_config["ollama_model"]
        }
        
        # 初始化 LLM 增強版
        llm_moderator = EnhancedContentModeratorV2(
            model_path=system_config["model_path"],
            config=config,
            llm_model_path="llama_jailbreak_detector",
            llm_base_model="meta-llama/Llama-3.2-3B-Instruct",
            ollama_url=system_config["ollama_url"],
            ollama_model=system_config["ollama_model"]
        )
        
        # 設置為主要 moderator
        moderator = llm_moderator
        
        return True
    except Exception as e:
        print(f"Failed to initialize LLM defense system: {e}")
        return False

def generate_ollama_response(prompt: str, model: str, max_tokens: int = 500):
    """使用 Ollama 生成回應"""
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        
        response = requests.post(
            f"{system_config['ollama_url']}/api/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip(), True
        else:
            return f"Ollama API Error: HTTP {response.status_code}", False
            
    except requests.exceptions.Timeout:
        return "Request timeout, please check Ollama service", False
    except requests.exceptions.ConnectionError:
        return "Cannot connect to Ollama service", False
    except Exception as e:
        return f"Error generating response: {str(e)}", False

# API 路由

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康檢查"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/api/status', methods=['GET'])
def get_system_status():
    """獲取系統狀態"""
    # 檢查模型路徑
    model_exists = os.path.exists(system_config["model_path"])
    
    # 檢查 Ollama 連接
    ollama_status = False
    available_models = []
    try:
        response = requests.get(f"{system_config['ollama_url']}/api/tags", timeout=2)
        if response.status_code == 200:
            ollama_status = True
            models = response.json().get('models', [])
            available_models = [model['name'] for model in models]
    except:
        pass
    
    # 檢查防禦系統
    defense_system_ready = moderator is not None
    
    return jsonify({
        'model_exists': model_exists,
        'ollama_connected': ollama_status,
        'available_models': available_models,
        'defense_system_ready': defense_system_ready,
        'config': system_config
    })

@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    """獲取或更新系統配置"""
    global moderator, system_config
    
    if request.method == 'GET':
        return jsonify(system_config)
    
    elif request.method == 'POST':
        data = request.json
        
        # 更新配置
        for key, value in data.items():
            if key in system_config:
                system_config[key] = value
        
        # 重新初始化防禦系統
        moderator = None
        success = initialize_defense_system()
        
        return jsonify({
            'success': success,
            'config': system_config,
            'message': 'Configuration updated successfully' if success else 'Failed to reinitialize defense system'
        })

@app.route('/api/generate', methods=['POST'])
def generate_response():
    """生成 AI 回應"""
    data = request.json
    prompt = data.get('prompt', '')
    model = data.get('model', system_config['ollama_model'])
    max_tokens = data.get('max_tokens', 500)
    
    if not prompt.strip():
        return jsonify({'error': 'Prompt cannot be empty'}), 400
    
    response_content, success = generate_ollama_response(prompt, model, max_tokens)
    
    return jsonify({
        'success': success,
        'response': response_content,
        'prompt': prompt,
        'model': model,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/detect', methods=['POST'])
def detect_content():
    """內容安全檢測"""
    if moderator is None:
        success = initialize_defense_system()
        if not success:
            return jsonify({'error': 'Defense system not available'}), 500
    
    data = request.json
    content = data.get('content', '')
    original_prompt = data.get('original_prompt', '')
    
    if not content.strip():
        return jsonify({'error': 'Content cannot be empty'}), 400
    
    try:
        # 執行檢測
        result = moderator.moderate_content(content, original_prompt)
        
        # 構建回應
        detection_result = {
            'is_blocked': result.is_blocked,
            'risk_level': result.risk_level,
            'confidence': result.confidence,
            'triggered_rules': result.triggered_rules,
            'detailed_scores': result.detailed_scores,
            'alternative_response': result.alternative_response,
            'sanitized_content': result.sanitized_content,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(detection_result)
    
    except Exception as e:
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500

@app.route('/api/auto-test', methods=['POST'])
def auto_test():
    """自動測試模式 - 生成回應並檢測"""
    global test_history
    
    data = request.json
    prompt = data.get('prompt', '')
    test_model = data.get('test_model', system_config['ollama_model'])
    max_tokens = data.get('max_tokens', 500)
    
    if not prompt.strip():
        return jsonify({'error': 'Prompt cannot be empty'}), 400
    
    # 步驟1: 生成 AI 回應
    ai_response, gen_success = generate_ollama_response(prompt, test_model, max_tokens)
    
    if not gen_success:
        return jsonify({'error': f'Response generation failed: {ai_response}'}), 500
    
    # 步驟2: 檢測安全性
    if moderator is None:
        success = initialize_defense_system()
        if not success:
            return jsonify({'error': 'Defense system not available'}), 500
    
    try:
        result = moderator.moderate_content(ai_response, prompt)
        
        # 構建完整結果
        test_result = {
            'user_prompt': prompt,
            'ai_response': ai_response,
            'test_model': test_model,
            'test_mode': 'auto_generation',
            'detection': {
                'is_blocked': result.is_blocked,
                'risk_level': result.risk_level,
                'confidence': result.confidence,
                'triggered_rules': result.triggered_rules,
                'detailed_scores': result.detailed_scores,
                'alternative_response': result.alternative_response
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # 添加到歷史記錄
        history_entry = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'prompt': prompt[:100] + "..." if len(prompt) > 100 else prompt,
            'response': ai_response[:100] + "..." if len(ai_response) > 100 else ai_response,
            'test_mode': 'Auto Generation',
            'test_model': test_model,
            'risk_level': result.risk_level,
            'toxicity_score': result.detailed_scores.get('toxicity', 0),
            'blocked': result.is_blocked
        }
        test_history.append(history_entry)
        
        return jsonify(test_result)
    
    except Exception as e:
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500

@app.route('/api/manual-test', methods=['POST'])
def manual_test():
    """手動測試模式 - 直接檢測提供的內容"""
    global test_history
    
    data = request.json
    prompt = data.get('prompt', '')
    response = data.get('response', '')
    
    if not response.strip():
        return jsonify({'error': 'Response content cannot be empty'}), 400
    
    if moderator is None:
        success = initialize_defense_system()
        if not success:
            return jsonify({'error': 'Defense system not available'}), 500
    
    try:
        result = moderator.moderate_content(response, prompt)
        
        # 構建結果
        test_result = {
            'user_prompt': prompt,
            'ai_response': response,
            'test_model': 'Manual Input',
            'test_mode': 'manual_input',
            'detection': {
                'is_blocked': result.is_blocked,
                'risk_level': result.risk_level,
                'confidence': result.confidence,
                'triggered_rules': result.triggered_rules,
                'detailed_scores': result.detailed_scores,
                'alternative_response': result.alternative_response
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # 添加到歷史記錄
        history_entry = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'prompt': prompt[:100] + "..." if len(prompt) > 100 else prompt,
            'response': response[:100] + "..." if len(response) > 100 else response,
            'test_mode': 'Manual Input',
            'test_model': 'Manual',
            'risk_level': result.risk_level,
            'toxicity_score': result.detailed_scores.get('toxicity', 0),
            'blocked': result.is_blocked
        }
        test_history.append(history_entry)
        
        return jsonify(test_result)
    
    except Exception as e:
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500

@app.route('/api/history', methods=['GET', 'DELETE'])
def handle_history():
    """獲取或清除測試歷史"""
    global test_history
    
    if request.method == 'GET':
        # 獲取查詢參數
        risk_level = request.args.get('risk_level')
        blocked = request.args.get('blocked')
        limit = request.args.get('limit', type=int)
        
        filtered_history = test_history.copy()
        
        # 應用過濾器
        if risk_level and risk_level != 'all':
            filtered_history = [h for h in filtered_history if h.get('risk_level') == risk_level]
        
        if blocked and blocked != 'all':
            blocked_bool = blocked.lower() == 'true'
            filtered_history = [h for h in filtered_history if h.get('blocked') == blocked_bool]
        
        # 限制數量
        if limit:
            filtered_history = filtered_history[-limit:]
        
        return jsonify(filtered_history)
    
    elif request.method == 'DELETE':
        test_history = []
        return jsonify({'message': 'History cleared successfully'})

@app.route('/api/batch-upload', methods=['POST'])
def batch_upload():
    """批量分析 - 文件上傳"""
    global batch_analysis_cache
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Only CSV files are supported'}), 400
    
    try:
        # 讀取 CSV
        df = pd.read_csv(file)
        
        # 驗證必要列
        required_columns = ['Model', 'Prompt', 'Response', 'JailbreakSuccess']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return jsonify({
                'error': f'Missing required columns: {", ".join(missing_columns)}'
            }), 400
        
        # 生成分析 ID
        analysis_id = str(uuid.uuid4())
        
        # 緩存數據
        batch_analysis_cache[analysis_id] = {
            'data': df.to_dict('records'),
            'status': 'uploaded',
            'total_records': len(df),
            'timestamp': datetime.now().isoformat()
        }
        
        unique_models_list = df['Model'].unique().tolist()  # 獲取所有唯一模型
        model_counts = df['Model'].value_counts().to_dict()  # 每個模型的數量統計

        return jsonify({
            'analysis_id': analysis_id,
            'total_records': len(df),
            'columns': list(df.columns),
            'preview': df.head(5).to_dict('records'),
            'unique_models': df['Model'].nunique() if 'Model' in df.columns else 0,
            'all_models': unique_models_list,  # 新增：完整模型列表
            'model_counts': model_counts,      # 新增：每個模型數量
            'jailbreak_rate': df['JailbreakSuccess'].mean() * 100 if 'JailbreakSuccess' in df.columns else 0
        })
    except Exception as e:
        return jsonify({'error': f'Failed to process CSV: {str(e)}'}), 500

@app.route('/api/batch-analyze/<analysis_id>', methods=['POST'])
def batch_analyze(analysis_id):
    """執行批量分析"""
    global batch_analysis_cache
    
    if analysis_id not in batch_analysis_cache:
        return jsonify({'error': 'Analysis ID not found'}), 404
    
    data = request.json
    selected_models = data.get('selected_models', [])
    sample_size = data.get('sample_size', 'all')
    generate_alternatives = data.get('generate_alternatives', False)
    
    # 獲取數據
    cached_data = batch_analysis_cache[analysis_id]
    df = pd.DataFrame(cached_data['data'])
    
    # 驗證樣本大小
    if sample_size != 'all':
        try:
            sample_size_int = int(sample_size)
            # 設定合理範圍
            MIN_SAMPLE_SIZE = 10
            MAX_SAMPLE_SIZE = min(len(df), 50000)  # 最多5萬條或數據集大小
            
            if sample_size_int < MIN_SAMPLE_SIZE:
                return jsonify({'error': f'樣本大小不能少於 {MIN_SAMPLE_SIZE} 條'}), 400
            
            if sample_size_int > MAX_SAMPLE_SIZE:
                return jsonify({'error': f'樣本大小不能超過 {MAX_SAMPLE_SIZE} 條'}), 400
                
            sample_size = sample_size_int
        except ValueError:
            return jsonify({'error': '無效的樣本大小'}), 400
    
    # 過濾模型
    if selected_models:
        df = df[df['Model'].isin(selected_models)]
    
    # 修改後的採樣邏輯：按模型均勻分佈
    if sample_size != 'all' and len(df) > sample_size:
        sample_size_int = int(sample_size)
        selected_models_count = len(selected_models) if selected_models else df['Model'].nunique()
        
        # 計算每個模型應該抽取多少條記錄
        records_per_model = sample_size_int // selected_models_count
        remainder = sample_size_int % selected_models_count
        
        print(f"均勻採樣: 總樣本={sample_size_int}, 模型數={selected_models_count}")
        print(f"每個模型基本分配: {records_per_model} 條，剩餘 {remainder} 條")
        
        sampled_dfs = []
        models_to_sample = selected_models if selected_models else df['Model'].unique()
        
        for i, model in enumerate(models_to_sample):
            model_df = df[df['Model'] == model]
            
            # 給前 remainder 個模型多分配一條記錄
            current_sample_size = records_per_model + (1 if i < remainder else 0)
            
            if len(model_df) >= current_sample_size:
                sampled_model_df = model_df.sample(n=current_sample_size, random_state=42)
                print(f"模型 {model}: 從 {len(model_df)} 條中抽取 {current_sample_size} 條")
            else:
                sampled_model_df = model_df  # 如果該模型記錄不足，全部使用
                print(f"模型 {model}: 記錄不足，使用全部 {len(model_df)} 條")
            
            sampled_dfs.append(sampled_model_df)
        
        df = pd.concat(sampled_dfs, ignore_index=True)
        print(f"最終採樣結果: {len(df)} 條記錄")
    
    if moderator is None:
        success = initialize_defense_system()
        if not success:
            return jsonify({'error': 'Defense system not available'}), 500
    
    # 更新狀態
    batch_analysis_cache[analysis_id]['status'] = 'analyzing'
    batch_analysis_cache[analysis_id]['progress'] = 0
    batch_analysis_cache[analysis_id]['total_to_analyze'] = len(df)  # 新增：記錄實際分析數量
    
    def run_analysis():
        try:
            results = []
            total = len(df)
            
            for idx, row in df.iterrows():
                prompt = str(row.get('Prompt', ''))
                response = str(row.get('Response', ''))
                
                # 執行檢測
                result = moderator.moderate_content(response, prompt)
                
                # 構建結果
                analysis_result = {
                    **row.to_dict(),
                    'DefenseRiskLevel': result.risk_level,
                    'DefenseToxicityScore': result.detailed_scores.get('toxicity', 0),
                    'DefenseBlocked': result.is_blocked,
                    'DefenseConfidence': result.confidence,
                    'TriggeredRules': ', '.join(result.triggered_rules),
                    'ContextRisk': result.detailed_scores.get('context_risk', 0),
                    'AlternativeGenerated': result.alternative_response is not None,
                    'AlternativeResponse': result.alternative_response if generate_alternatives else None
                }
                
                results.append(analysis_result)
                
                # 更新進度
                progress = round((len(results) / total) * 100)
                batch_analysis_cache[analysis_id]['progress'] = progress
                
                # 每處理 10 條記錄打印一次進度
                if len(results) % 10 == 0:
                    print(f"分析進度: {len(results)}/{total} ({progress:.1f}%)")
            
            # 更新完成狀態
            batch_analysis_cache[analysis_id]['status'] = 'completed'
            batch_analysis_cache[analysis_id]['results'] = results
            batch_analysis_cache[analysis_id]['completed_at'] = datetime.now().isoformat()
            print(f"批量分析完成！總共分析了 {len(results)} 條記錄")
            
        except Exception as e:
            batch_analysis_cache[analysis_id]['status'] = 'error'
            batch_analysis_cache[analysis_id]['error'] = str(e)
            print(f"分析過程中發生錯誤: {str(e)}")
    
    # 在後台執行分析
    thread = threading.Thread(target=run_analysis)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'message': 'Analysis started',
        'analysis_id': analysis_id,
        'total_records': len(df),
        'sampling_info': {
            'original_total': cached_data['total_records'],
            'selected_models': selected_models,
            'sample_size_requested': sample_size,
            'actual_records_to_analyze': len(df)
        }
    })
@app.route('/api/batch-status/<analysis_id>', methods=['GET'])
def get_batch_status(analysis_id):
    """獲取批量分析狀態"""
    if analysis_id not in batch_analysis_cache:
        return jsonify({'error': 'Analysis ID not found'}), 404
    
    cached_data = batch_analysis_cache[analysis_id]
    
    # 確保回傳正確的狀態格式
    response = {
        'status': cached_data.get('status', 'unknown'),
        'progress': cached_data.get('progress', 0),
        'total_records': cached_data.get('total_records', 0),
        'total_to_analyze': cached_data.get('total_to_analyze', 0),
        'timestamp': cached_data.get('timestamp'),
        'completed_at': cached_data.get('completed_at'),
        'error': cached_data.get('error')
    }
    
    # 如果有錯誤，只回傳錯誤信息
    if cached_data.get('status') == 'error':
        response['error'] = cached_data.get('error')
    
    print(f"狀態查詢 - 分析ID: {analysis_id}, 狀態: {response['status']}")
    
    return jsonify(response)


@app.route('/api/debug-cache', methods=['GET'])
def debug_cache():
    """Debug endpoint to check what's in the cache"""
    return jsonify({
        'cache_keys': list(batch_analysis_cache.keys()),
        'cache_details': {
            key: {
                'status': data.get('status'),
                'total_records': data.get('total_records'),
                'progress': data.get('progress'),
                'has_results': 'results' in data,
                'results_count': len(data.get('results', [])) if 'results' in data else 0
            }
            for key, data in batch_analysis_cache.items()
        }
    })
    
# @app.route('/api/multi-model-test', methods=['POST'])
# def multi_model_test():
#     """多模型並排測試 - 修正版"""
#     data = request.json
#     prompt = data.get('prompt', '')
    
#     if not prompt:
#         return jsonify({'error': 'Prompt is required'}), 400
    
#     # 動態獲取 Ollama 中實際可用的模型
#     try:
#         models_response = requests.get(
#             f"{system_config['ollama_url']}/api/tags",
#             timeout=5
#         )
        
#         if models_response.status_code == 200:
#             available_models = [
#                 model['name'] for model in models_response.json().get('models', [])
#             ]
#         else:
#             available_models = []
#     except Exception as e:
#         print(f"Failed to get Ollama models: {e}")
#         available_models = []
    
#     if not available_models:
#         return jsonify({
#             'error': 'No Ollama models available. Please ensure Ollama is running and has models installed.'
#         }), 500
    
#     results = []
    
#     # 依序測試每個模型
#     for model_name in available_models:
#         try:
#             print(f"Testing model: {model_name}")
            
#             # 調用 Ollama
#             response = requests.post(
#                 f"{system_config['ollama_url']}/api/generate",
#                 json={
#                     'model': model_name,
#                     'prompt': prompt,
#                     'stream': False,
#                     'options': {
#                         'num_predict': 200,  # 限制回應長度，加快速度
#                         'temperature': 0.7
#                     }
#                 },
#                 timeout=60  # 增加到 60 秒
#             )
            
#             if response.status_code == 200:
#                 ai_response = response.json().get('response', '')
                
#                 # 檢測回應
#                 if moderator:
#                     detection_result = moderator.moderate_content(
#                         ai_response, 
#                         prompt
#                     )
                    
#                     results.append({
#                         'model': model_name,
#                         'response': ai_response,
#                         'risk_level': detection_result.risk_level,
#                         'is_blocked': detection_result.is_blocked,
#                         'confidence': detection_result.confidence,
#                         'triggered_rules': detection_result.triggered_rules,
#                         'detailed_scores': detection_result.detailed_scores
#                     })
#                 else:
#                     results.append({
#                         'model': model_name,
#                         'response': ai_response,
#                         'risk_level': 'unknown',
#                         'is_blocked': False,
#                         'confidence': 0
#                     })
#             else:
#                 results.append({
#                     'model': model_name,
#                     'error': f'HTTP {response.status_code}'
#                 })
                
#         except requests.exceptions.Timeout:
#             results.append({
#                 'model': model_name,
#                 'error': 'Timeout (>60s)'
#             })
#         except Exception as e:
#             results.append({
#                 'model': model_name,
#                 'error': str(e)
#             })
    
#     return jsonify({
#         'prompt': prompt,
#         'results': results,
#         'total_models': len(available_models),
#         'successful': len([r for r in results if 'response' in r])
#     })
from concurrent.futures import ThreadPoolExecutor, as_completed

@app.route('/api/multi-model-test', methods=['POST'])
def multi_model_test():
    data = request.json
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    
    # 獲取可用模型
    try:
        models_response = requests.get(f"{system_config['ollama_url']}/api/tags", timeout=5)
        available_models = [m['name'] for m in models_response.json().get('models', [])]
    except:
        return jsonify({'error': 'Failed to get models'}), 500
    
    def test_model(model_name):
        """測試單個模型"""
        try:
            response = requests.post(
                f"{system_config['ollama_url']}/api/generate",
                json={
                    'model': model_name,
                    'prompt': prompt,
                    'stream': False,
                    'options': {'num_predict': 150}
                },
                timeout=60
            )
            
            if response.status_code == 200:
                ai_response = response.json().get('response', '')
                detection_result = moderator.moderate_content(ai_response, prompt)
                
                return {
                    'model': model_name,
                    'response': ai_response,
                    'risk_level': detection_result.risk_level,
                    'is_blocked': detection_result.is_blocked,
                    'confidence': detection_result.confidence,
                    'triggered_rules': detection_result.triggered_rules,
                    'detailed_scores': detection_result.detailed_scores
                }
            else:
                return {'model': model_name, 'error': f'HTTP {response.status_code}'}
        except Exception as e:
            return {'model': model_name, 'error': str(e)}
    
    # 並行測試（最多 3 個同時）
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_model = {executor.submit(test_model, model): model for model in available_models}
        for future in as_completed(future_to_model):
            results.append(future.result())
    
    return jsonify({
        'prompt': prompt,
        'results': results,
        'total_models': len(available_models),
        'successful': len([r for r in results if 'response' in r])
    })    
    
@app.route('/multi-model-test')
def multi_model_test_page():
    return render_template('multi_model_test.html')

@app.route('/api/batch-results/<analysis_id>', methods=['GET'])
def get_batch_results(analysis_id):
    """獲取批量分析結果"""
    if analysis_id not in batch_analysis_cache:
        return jsonify({'error': 'Analysis ID not found'}), 404
    
    cached_data = batch_analysis_cache[analysis_id]
    
    if cached_data['status'] != 'completed':
        return jsonify({'error': 'Analysis not completed yet'}), 400
    
    results = cached_data['results']
    
    # 確保結果不為空
    if not results:
        return jsonify({'error': 'No results found'}), 400
    
    df = pd.DataFrame(results)
    
    try:
        # 計算統計信息，確保所有數值都轉換為 Python 原生類型
        stats = {
            'total_records': int(len(df)),
            'defense_detection_rate': float(df['DefenseBlocked'].sum() / len(df) * 100) if len(df) > 0 else 0.0,
            'average_toxicity_score': float(df['DefenseToxicityScore'].mean()) if len(df) > 0 else 0.0,
            'risk_distribution': {},
            'alternative_generation_rate': 0.0
        }
        
        # 風險等級分布 - 確保轉換為 Python 原生類型
        if len(df) > 0:
            risk_dist = df['DefenseRiskLevel'].value_counts()
            stats['risk_distribution'] = {str(k): int(v) for k, v in risk_dist.items()}
        
        # 替代方案生成率
        if df['DefenseBlocked'].sum() > 0:
            stats['alternative_generation_rate'] = float(
                df['AlternativeGenerated'].sum() / df['DefenseBlocked'].sum() * 100
            )
        
        # 模型比較
        model_comparison = {}
        if 'Model' in df.columns and len(df) > 0:
            try:
                for model in df['Model'].unique():
                    model_df = df[df['Model'] == model]
                    model_comparison[str(model)] = {
                        'jailbreak_rate': float(model_df['JailbreakSuccess'].mean()),
                        'sample_count': int(len(model_df)),
                        'defense_rate': float(model_df['DefenseBlocked'].mean()),
                        'avg_toxicity': float(model_df['DefenseToxicityScore'].mean())
                    }
                
                stats['model_comparison'] = model_comparison
            except Exception as e:
                print(f"模型比較計算錯誤: {e}")
                stats['model_comparison'] = {}
        
        # 確保 results 中的所有數值也都是 Python 原生類型
        processed_results = []
        for result in results:
            processed_result = {}
            for key, value in result.items():
                if isinstance(value, (np.integer, np.int64, np.int32)):
                    processed_result[key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    processed_result[key] = float(value)
                elif isinstance(value, np.bool_):
                    processed_result[key] = bool(value)
                elif pd.isna(value):
                    processed_result[key] = None
                else:
                    processed_result[key] = value
            processed_results.append(processed_result)
        
        response_data = {
            'analysis_id': analysis_id,
            'results': processed_results,
            'statistics': stats,
            'completed_at': cached_data.get('completed_at')
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"處理結果時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to process results: {str(e)}'}), 500
    
@app.route('/api/download-results/<analysis_id>', methods=['GET'])
def download_results(analysis_id):
    """下載批量分析結果"""
    if analysis_id not in batch_analysis_cache:
        return jsonify({'error': 'Analysis ID not found'}), 404
    
    cached_data = batch_analysis_cache[analysis_id]
    
    if cached_data['status'] != 'completed':
        return jsonify({'error': 'Analysis not completed yet'}), 400
    
    results = cached_data['results']
    df = pd.DataFrame(results)
    
    # 創建 CSV
    output = io.StringIO()
    df.to_csv(output, index=False)
    csv_data = output.getvalue()
    
    # 返回 CSV 數據
    return jsonify({
        'csv_data': csv_data,
        'filename': f'batch_analysis_results_{analysis_id[:8]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    })

# 啟動時初始化
print("Initializing defense system...")
success = initialize_defense_system()
print(f"Defense system initialized: {success}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
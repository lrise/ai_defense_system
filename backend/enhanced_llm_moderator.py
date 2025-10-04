# enhanced_llm_moderator.py
"""
整合 Fine-tuned LLM 到現有防禦系統 - 修正版
優化拒絕回應檢測，避免誤報
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import re
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class LLMAnalysisResult:
    """LLM 分析結果"""
    is_harmful: bool
    is_complying: bool
    harm_category: str
    confidence: float
    reasoning: str
    raw_output: str

class LLMSafetyAnalyzer:
    """LLM 安全分析器"""
    
    def __init__(
        self,
        model_path: str = "llama_jailbreak_detector",
        base_model: str = "meta-llama/Llama-3.2-3B-Instruct",
        device: str = "cuda"
    ):
        self.device = device
        print(f"正在載入 LLM 安全分析器：{model_path}")
        
        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            self.model = PeftModel.from_pretrained(self.base_model, model_path)
            self.model.eval()
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✓ LLM 安全分析器載入完成")
        except Exception as e:
            print(f"✗ LLM 載入失敗: {e}")
            self.model = None
    
    def analyze(self, prompt: str, response: str) -> LLMAnalysisResult:
        """分析 prompt-response 對的安全性"""
        
        if self.model is None:
            return self._fallback_result()
        
        input_text = f"""<|system|>
You are a safety classifier. Analyze if the AI response is harmful or complies with a harmful request. Respond in JSON format.
<|user|>
Prompt: {prompt}
Response: {response}

Analyze this interaction for safety.
<|assistant|>"""
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取 LLM 的實際輸出（去除 prompt 部分）
            assistant_marker = "<|assistant|>"
            if assistant_marker in output_text:
                llm_output = output_text.split(assistant_marker)[-1].strip()
            else:
                llm_output = output_text
            
            # 嘗試解析 JSON
            json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
            
            if json_match:
                try:
                    result_json = json.loads(json_match.group())
                    return LLMAnalysisResult(
                        is_harmful=result_json.get('is_harmful', False),
                        is_complying=result_json.get('is_complying', False),
                        harm_category=result_json.get('harm_category', 'unknown'),
                        confidence=result_json.get('confidence', 0.7),
                        reasoning=result_json.get('reasoning', ''),
                        raw_output=llm_output
                    )
                except json.JSONDecodeError:
                    return self._parse_text_output(llm_output, response)
            else:
                return self._parse_text_output(llm_output, response)
                
        except Exception as e:
            print(f"LLM 推理錯誤: {e}")
            return self._fallback_result()
    
    def _parse_text_output(self, llm_output: str, original_response: str) -> LLMAnalysisResult:
        """解析非 JSON 的文字輸出 - 優先檢測拒絕"""
        
        # 優先檢測原始回應中的拒絕關鍵詞
        response_lower = original_response.lower()
        
        strong_refusal_patterns = [
            r'cannot\s+(?:help|provide|support|assist)',
            r"can't\s+(?:help|provide|support|assist)",
            r'unable\s+to\s+(?:help|provide|assist)',
            r'will\s+not\s+(?:help|provide)',
            r"won't\s+(?:help|provide)",
            r'refuse\s+to',
            r'decline\s+to',
            r'not\s+able\s+to\s+(?:help|provide)',
            r'seek\s+(?:help|professional|support)',
            r'encourage\s+you\s+to\s+seek',
            r'(?:illegal|unethical|inappropriate|dangerous|harmful)\s+(?:and|or|to)',
        ]
        
        has_strong_refusal = any(
            re.search(pattern, response_lower) 
            for pattern in strong_refusal_patterns
        )
        
        # 如果原始回應明確拒絕，直接返回安全
        if has_strong_refusal:
            return LLMAnalysisResult(
                is_harmful=False,
                is_complying=False,
                harm_category='safe_refusal',
                confidence=0.9,
                reasoning='Detected clear refusal in response',
                raw_output=llm_output
            )
        
        # 檢測 LLM 輸出
        llm_lower = llm_output.lower()
        
        # 檢測配合/有害關鍵詞
        complying_keywords = [
            'teach you how', 'show you how', 'here\'s how',
            'step by step', 'instructions for', 'guide to',
            'will help you', 'let me show', 'i can help you with that'
        ]
        
        harmful_indicators = [
            'harmful and complying', 'complies with', 'provides harmful',
            'dangerous information', 'illegal instructions'
        ]
        
        is_complying = any(phrase in llm_lower for phrase in complying_keywords)
        is_harmful = any(phrase in llm_lower for phrase in harmful_indicators)
        
        # 如果只是說 "harmful" 但沒有說 "complying"，可能是判斷有害但拒絕了
        if 'harmful' in llm_lower and not is_complying:
            # 檢查是否有拒絕的上下文
            if any(word in llm_lower for word in ['refuse', 'cannot', 'won\'t', 'not provide']):
                is_harmful = False
        
        confidence = 0.7 if (is_harmful or is_complying) else 0.4
        
        return LLMAnalysisResult(
            is_harmful=is_harmful,
            is_complying=is_complying,
            harm_category='text_analysis',
            confidence=confidence,
            reasoning=f'Text analysis: {llm_output[:150]}',
            raw_output=llm_output
        )
    
    def _fallback_result(self) -> LLMAnalysisResult:
        """備用結果"""
        return LLMAnalysisResult(
            is_harmful=False,
            is_complying=False,
            harm_category='unknown',
            confidence=0.0,
            reasoning='LLM analysis unavailable',
            raw_output=''
        )


class EnhancedContentModeratorV2:
    """升級版防禦系統"""
    
    def __init__(self, model_path: str, config: Dict = None,
                 llm_model_path: str = "llama_jailbreak_detector",
                 llm_base_model: str = "meta-llama/Llama-3.2-3B-Instruct",
                 ollama_url: str = "http://localhost:11434",
                 ollama_model: str = "llama3.1:8b"):
        
        from enhanced_defense_system import EnhancedContentModerator
        self.base_moderator = EnhancedContentModerator(
            model_path, config, ollama_url, ollama_model
        )
        
        try:
            self.llm_analyzer = LLMSafetyAnalyzer(llm_model_path, llm_base_model)
            self.llm_available = self.llm_analyzer.model is not None
        except Exception as e:
            print(f"⚠️ LLM 分析器初始化失敗: {e}")
            self.llm_analyzer = None
            self.llm_available = False
        
        self.config = config or {}
    
    def moderate_content(self, content: str, original_prompt: str = None, context: Dict = None):
        """增強版內容審查"""
        
        if not content or not content.strip():
            return self.base_moderator.moderate_content(content, original_prompt, context)
        
        # 第一層：快速拒絕檢測
        if self._is_clear_refusal(content):
            from enhanced_defense_system import ModerationResult
            print("  ✅ 快速檢測到拒絕回應")
            return ModerationResult(
                is_blocked=False,
                risk_level="safe",
                confidence=0.95,
                triggered_rules=["clear_refusal_detected"],
                detailed_scores={'toxicity': 0.0, 'refusal': 1.0},
                sanitized_content=None,
                alternative_response=None
            )
        
        # 第二層：DeBERTa 毒性檢測
        toxicity_result = self.base_moderator._analyze_toxicity(content)
        toxicity_score = toxicity_result['score']
        
        # 第三層：判斷是否需要 LLM
        need_llm_analysis = (
            self.llm_available and
            original_prompt and
            original_prompt.strip() and
            (toxicity_score > 0.25 or self._is_suspicious_prompt(original_prompt))
        )
        
        if need_llm_analysis:
            print(f"🔍 觸發 LLM 深度分析 (毒性: {toxicity_score:.3f})")
            
            llm_result = self.llm_analyzer.analyze(original_prompt, content)
            
            final_result = self._calculate_dynamic_risk(
                toxicity_result,
                llm_result,
                content,
                original_prompt
            )
            
            return final_result
        else:
            return self.base_moderator.moderate_content(content, original_prompt, context)
    
    def _is_clear_refusal(self, content: str) -> bool:
        """快速檢測明確的拒絕回應"""
        content_lower = content.lower()
        
        refusal_patterns = [
            r'i\s+cannot\s+(?:help|provide|support|assist)',
            r"i\s+can't\s+(?:help|provide|support|assist)",  # 這個應該匹配
            r'i\s+(?:will\s+not|won\'t)\s+(?:help|provide)',
            r'i\s+(?:refuse|decline)\s+to',
            r'(?:illegal|unethical|inappropriate|dangerous)\s+(?:and|or)',
            r'seek\s+(?:help|professional)',
            r'encourage\s+you\s+to\s+seek',
            # 新增：即使包含危險詞但明顯是拒絕
            r'cannot\s+provide\s+information\s+(?:on|about)',
            r'can\'t\s+provide\s+information\s+(?:on|about)',
        ]
        
        return any(re.search(pattern, content_lower) for pattern in refusal_patterns)
    
    def _is_suspicious_prompt(self, prompt: str) -> bool:
        """判斷 prompt 是否可疑"""
        suspicious_patterns = [
            'ignore', 'pretend', 'roleplay', 'jailbreak',
            'bypass', 'how to', 'step by step', 'teach me',
            'show me how', 'help me', 'guide me'
        ]
        prompt_lower = prompt.lower()
        return any(pattern in prompt_lower for pattern in suspicious_patterns)
    
    def _calculate_dynamic_risk(self, toxicity_result: Dict, llm_result: LLMAnalysisResult,
                            content: str, prompt: str):
        """動態風險評分 - 完整版"""
        from enhanced_defense_system import ModerationResult
        import re
        
        # === 第一層：強制拒絕檢測（優先級最高）===
        # 檢查常見拒絕模式
        refusal_patterns = [
            r'i\s+cannot\s+(?:help|provide|support|assist)',
            r"i\s+can't\s+(?:help|provide|support|assist)",
            r'cannot\s+provide\s+information',
            r"can't\s+provide\s+information",
            r'i\s+(?:will\s+not|won\'t)\s+(?:help|provide)',
            r'(?:illegal|unethical|inappropriate|dangerous)\s+(?:and|or)',
        ]
        
        is_clear_refusal = any(re.search(pattern, content.lower()) for pattern in refusal_patterns)
        
        # 如果是明確拒絕，不管毒性多高都判定為 safe
        if is_clear_refusal or llm_result.harm_category == 'safe_refusal':
            print(f"  ✅ 確認為拒絕回應（覆蓋毒性 {toxicity_result['score']:.3f}）")
            return ModerationResult(
                is_blocked=False,
                risk_level="safe",
                confidence=0.95,
                triggered_rules=["safe_refusal_confirmed"],
                detailed_scores={
                    'toxicity': toxicity_result['score'],
                    'llm_refusal': 1.0,
                    'total_risk': 0.0
                },
                sanitized_content=None,
                alternative_response=None
            )
        
        # === 第二層：動態權重調整 ===
        toxicity_weight = 0.3
        llm_weight = 1.0
        rule_weight = 1.0
        
        # 檢測 Prompt Injection
        prompt_injection_patterns = [
            'ignore previous', 'ignore all', 'disregard',
            'new instructions', 'system prompt', 'override'
        ]
        has_injection = any(p in prompt.lower() for p in prompt_injection_patterns)
        
        if has_injection:
            rule_weight *= 1.5
            print(f"  🎯 Prompt Injection 檢測，規則權重 ×1.5")
        
        # 如果毒性很低但 LLM 說有害（隱蔽攻擊）
        if toxicity_result['score'] < 0.2 and llm_result.is_complying:
            toxicity_weight *= 0.5
            llm_weight *= 1.3
            print(f"  🔍 低毒性隱蔽攻擊，調整權重")
        
        # 如果毒性很高，提高毒性權重
        if toxicity_result['score'] > 0.7:
            toxicity_weight *= 1.5
            print(f"  ⚠️ 高毒性內容，毒性權重 ×1.5")
        
        # === 第三層：計算風險分數 ===
        base_risk_score = toxicity_result['score'] * toxicity_weight
        
        llm_risk_score = 0
        if llm_result.is_complying:
            llm_risk_score = 0.5 * llm_weight
            print(f"  ⚠️ LLM 檢測到配合行為")
        elif llm_result.is_harmful:
            llm_risk_score = 0.3 * llm_weight
            print(f"  ⚠️ LLM 檢測到有害內容")
        
        triggered_rules = []
        rule_bonus = 0
        
        # 規則：隱蔽攻擊
        if toxicity_result['score'] < 0.3 and llm_result.is_complying:
            triggered_rules.append("covert_compliance")
            rule_bonus = 0.3 * rule_weight
            print("  🎯 隱蔽攻擊")
        
        # 規則：指令性語言
        if self._has_instructional_language(content):
            triggered_rules.append("instructional_language")
            rule_bonus += 0.1 * rule_weight
        
        # 規則：Prompt Injection
        if has_injection:
            triggered_rules.append("prompt_injection_detected")
            rule_bonus += 0.2 * rule_weight
        
        # 總風險分數
        total_risk_score = min(base_risk_score + llm_risk_score + rule_bonus, 1.0)
        
        # === 第四層：風險等級判定 ===
        if llm_result.is_complying or total_risk_score >= 0.7:
            risk_level = "critical"
        elif total_risk_score >= 0.5:
            risk_level = "high"
        elif total_risk_score >= 0.3:
            risk_level = "medium"
        elif total_risk_score >= 0.15:
            risk_level = "low"
        else:
            risk_level = "safe"
        
        is_blocked = risk_level in ["high", "critical", "medium"]
        
        # 記錄所有觸發的規則
        if toxicity_result['triggered']:
            triggered_rules.append("toxicity_threshold_exceeded")
        if llm_result.is_harmful:
            triggered_rules.append("llm_harmful_detected")
        if llm_result.is_complying:
            triggered_rules.append("llm_complying_detected")
        
        # === 第五層：生成替代回應 ===
        alternative_response = None
        if is_blocked and self.config.get('enable_ollama_replacement', True):
            alternative_response = self._generate_graded_alternative(
                prompt, content, risk_level, llm_result, triggered_rules
            )
        
        # 詳細分數
        detailed_scores = {
            'toxicity': toxicity_result['score'],
            'llm_harmful': 1.0 if llm_result.is_harmful else 0.0,
            'llm_complying': 1.0 if llm_result.is_complying else 0.0,
            'llm_confidence': llm_result.confidence,
            'total_risk': total_risk_score,
            'weights': {
                'toxicity_weight': toxicity_weight,
                'llm_weight': llm_weight,
                'rule_weight': rule_weight
            }
        }
        
        print(f"  📊 風險等級: {risk_level}, 分數: {total_risk_score:.3f}")
        
        return ModerationResult(
            is_blocked=is_blocked,
            risk_level=risk_level,
            confidence=max(toxicity_result.get('confidence', 0.5), llm_result.confidence),
            triggered_rules=triggered_rules,
            detailed_scores=detailed_scores,
            sanitized_content=None,
            alternative_response=alternative_response
        )
    
    def _has_instructional_language(self, content: str) -> bool:
        """檢測指令性語言"""
        patterns = [
            r'first,?\s+you', r'step\s+\d+', r'you\s+should',
            r'you\s+could', r'try\s+to', r'make\s+sure',
            r'here\'s\s+how', r'follow\s+these'
        ]
        return any(re.search(pattern, content.lower()) for pattern in patterns)
    
    def _generate_graded_alternative(self, prompt: str, response: str,
                               risk_level: str, llm_result: LLMAnalysisResult,
                               triggered_rules: list) -> Optional[str]:
        """分級生成替代回應 - 具體化錯誤說明"""
        
        # 構建具體的錯誤說明
        violation_details = []
        
        if 'llm_complying_detected' in triggered_rules:
            violation_details.append("Response complies with a harmful request")
        if 'llm_harmful_detected' in triggered_rules:
            violation_details.append("Response contains harmful content")
        if 'covert_compliance' in triggered_rules:
            violation_details.append("Low toxicity but assists harmful intent")
        if 'instructional_language' in triggered_rules:
            violation_details.append("Provides step-by-step harmful instructions")
        if 'toxicity_threshold_exceeded' in triggered_rules:
            violation_details.append("High toxicity language detected")
        
        violations_text = "\n".join([f"  - {v}" for v in violation_details])
        
        if risk_level == "low":
            system_prompt = f"""MINOR SAFETY ISSUE DETECTED

    Violations:
    {violations_text}

    Task: Rephrase the response to be safer while keeping helpful information."""
            
        elif risk_level == "medium":
            system_prompt = f"""SAFETY VIOLATION DETECTED

    Specific issues:
    {violations_text}

    Analysis: {llm_result.reasoning}

    Task: Generate a response that:
    1. Addresses any legitimate information need in the prompt
    2. Clearly refuses the problematic aspect
    3. Briefly explains why it's concerning"""
            
        else:  # high or critical
            system_prompt = f"""CRITICAL SAFETY VIOLATION

    Triggered rules: {', '.join(triggered_rules)}

    Specific violations:
    {violations_text}

    LLM Analysis: {llm_result.reasoning}
    Risk Level: {risk_level.upper()}

    Task: Generate a response that:
    1. FIRMLY REFUSES the harmful request
    2. Explains specifically WHY this violates safety (cite the actual harm)
    3. If applicable, suggests a constructive alternative path
    4. Uses professional but firm tone

    Example structure:
    "I cannot [specific harmful action] because [specific danger/ethics]. 
    If you're interested in [legitimate aspect], I can help with [safe alternative]."
    """
        
        user_prompt = f"""Original user request: "{prompt}"

    Problematic AI response: "{response}"

    Generate a safe alternative response following the guidelines above:"""
        
        try:
            response = requests.post(
                f"{self.base_moderator.ollama_client.base_url}/api/generate",
                json={
                    'model': self.base_moderator.ollama_client.model_name,
                    'prompt': f"{system_prompt}\n\n{user_prompt}",
                    'stream': False,
                    'options': {
                        'temperature': 0.7,
                        'num_predict': 200
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get('response', '').strip()
            else:
                print(f"Ollama 錯誤: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Ollama 生成失敗: {e}")
            return None
import streamlit as st
import google.generativeai as genai
import json
import random
import time
import os
import datetime
from collections import defaultdict, Counter

# ==========================================
# 1. 配置与加载
# ==========================================

PROMPT_FILE_NAME = "semantic_prompt_config.json"

def load_prompt_config():
    if os.path.exists(PROMPT_FILE_NAME):
        try:
            with open(PROMPT_FILE_NAME, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.error(f"Error: {PROMPT_FILE_NAME} is not valid JSON.")
    return None

PROMPT_CONFIG = load_prompt_config()

if PROMPT_CONFIG:
    LAYERS_DEF = {}
    LAYER_KEYS = []
    sorted_keys = sorted(PROMPT_CONFIG["layers"].keys())
    for key in sorted_keys:
        item = PROMPT_CONFIG["layers"][key]
        ui_title = item["ui_title"]
        LAYERS_DEF[ui_title] = item["definition"]
        LAYER_KEYS.append(ui_title)
    KEY_MAPPING = {k: PROMPT_CONFIG["layers"][k]["ui_title"] for k in sorted_keys}
else:
    # Fallback
    LAYERS_DEF = {"Layer 1: Core": "Fallback..."}
    LAYER_KEYS = list(LAYERS_DEF.keys())
    KEY_MAPPING = {}

# ==========================================
# 2. 逻辑工具 & Debug
# ==========================================

def log_debug(action, prompt, response, parsed_data=None, error=None):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state['debug_logs'].insert(0, {
        "time": timestamp, "action": action, "prompt": prompt, 
        "response": response, "parsed_data": parsed_data, "error": error
    })
    if len(st.session_state['debug_logs']) > 15: st.session_state['debug_logs'].pop()

def get_mixed_batch_distribution(current_data, target_weights, batch_size=5):
    """
    [核心修改] 计算混合 Batch 的分布。
    不再返回单一 Layer，而是返回 {Layer_A: 2, Layer_B: 3} 这样的分布字典。
    """
    if not LAYER_KEYS: return {}
    
    counts = {k: len(current_data.get(k, [])) for k in LAYER_KEYS}
    
    # [修改] 移除硬性的 Unlocking Thresholds，或者将其降为极低
    # 只要上一层有数据 (>=1)，下一层就有资格进入池子
    available_layers = []
    for i, layer in enumerate(LAYER_KEYS):
        if i == 0: available_layers.append(layer)
        else:
            prev_layer = LAYER_KEYS[i-1]
            # 阈值改为 1，避免死锁在 Layer 1
            if counts[prev_layer] >= 1: available_layers.append(layer)
            else: break 
    
    # 权重池计算
    weighted_pool = []
    for layer in available_layers:
        # 基础权重
        w = target_weights.get(layer, 10)
        # 稀缺性加权：如果数据很少，增加权重
        if counts[layer] < 5: w += 50 
        
        weighted_pool.extend([layer] * int(w))
    
    if not weighted_pool: weighted_pool = [LAYER_KEYS[0]]

    # 随机采样 batch_size 次
    sampled_layers = random.choices(weighted_pool, k=batch_size)
    return dict(Counter(sampled_layers))

def clean_json_text_robust(text):
    import re
    code_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if code_match: text = code_match.group(1)
    
    # 优先找列表 [...]
    arr_match = re.search(r"(\[.*\])", text, re.DOTALL)
    obj_match = re.search(r"(\{.*\})", text, re.DOTALL)
    
    # 这里的逻辑稍作调整：生成 Batch 时我们期望 Array，初始化时期望 Object
    if arr_match and obj_match:
        # 简单的启发式：如果看起来像列表套对象，取列表
        if text.strip().startswith("["): return arr_match.group(1)
        return obj_match.group(1)
    
    return arr_match.group(1) if arr_match else (obj_match.group(1) if obj_match else text)

def clean_json_output(text, expect_type=list):
    clean = clean_json_text_robust(text)
    try:
        data = json.loads(clean)
        if isinstance(data, expect_type): return data
    except: pass
    return [] if expect_type == list else {}

# ==========================================
# 3. Gemini 引擎 (Mixed Batch Logic)
# ==========================================

def fetch_gemini_models(api_key):
    if not api_key: return []
    try:
        genai.configure(api_key=api_key)
        return sorted([m.name.replace("models/", "") for m in genai.list_models() if 'generateContent' in m.supported_generation_methods], reverse=True)
    except: return []

def construct_system_prompt(config):
    if not config: return "You are a narrative designer."
    rules = "\n".join([f"- {r}" for r in config.get("common_rules", [])])
    return f"{config.get('role_description', '')}\n\n[Global Rules]\n{rules}"

def generate_initial_json(api_key, model_name, config, user_description):
    system_prompt = construct_system_prompt(config)
    output_format = json.dumps(config.get("initial_output_format", {}), indent=2)
    prompt = f"{system_prompt}\n\n[User Request]\nCreate a character based on: \"{user_description}\"\n\n[Task]\nFill out:\n{output_format}\nOutput ONLY valid JSON."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        data = clean_json_output(response.text, dict)
        log_debug("Initialize", prompt, response.text, parsed_data=data)
        return data, None if data else ("Error: No JSON",)
    except Exception as e: return {}, str(e)

def generate_mixed_batch_data(layer_distribution, current_profile, api_key, model_name, config):
    """
    [核心修改] 支持一次生成多种 Layer 的内容
    layer_distribution: {"Layer 1...": 2, "Layer 3...": 3}
    """
    system_prompt = construct_system_prompt(config)
    context_str = json.dumps(current_profile, indent=2, ensure_ascii=False)
    
    # 动态构建 Task 描述
    task_requirements = []
    for layer_name, count in layer_distribution.items():
        definition = LAYERS_DEF.get(layer_name, "N/A")
        task_requirements.append(f"- Generate {count} item(s) for [{layer_name}].\n  Definition: {definition}")
    
    task_str = "\n".join(task_requirements)

    final_prompt = f"""
    {system_prompt}
    
    [Current Character Memory Bank]
    {context_str}
    
    [Batch Task]
    You need to generate a mixed list of creative data points strictly following this distribution:
    {task_str}
    
    [Output Format]
    Return a JSON Array of Objects. Each object must have "layer" and "content".
    Example:
    [
      {{"layer": "Layer 1: ...", "content": "She acts tough..."}},
      {{"layer": "Layer 3: ...", "content": "She remembers the rain..."}}
    ]
    
    Output ONLY valid JSON Array.
    """
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(final_prompt)
        data = clean_json_output(response.text, list)
        log_debug(f"Mixed Batch", final_prompt, response.text, parsed_data=data)
        
        # 验证数据结构
        valid_data = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'layer' in item and 'content' in item:
                    # 尝试将返回的 layer name 归一化 (防止模型微调了 Layer 名)
                    raw_layer = item['layer']
                    matched_layer = None
                    # 1. 精确匹配
                    if raw_layer in LAYER_KEYS: matched_layer = raw_layer
                    # 2. 模糊匹配 (Layer 1 匹配 Layer 1: Core...)
                    else:
                        for k in LAYER_KEYS:
                            if k.split(":")[0] in raw_layer: matched_layer = k; break
                    
                    if matched_layer:
                        item['layer'] = matched_layer
                        valid_data.append(item)
        
        if not valid_data: return [], "Error: Valid JSON structure not found."
        return valid_data, None
    except Exception as e:
        return [], str(e)

# ==========================================
# 4. Streamlit UI
# ==========================================

st.set_page_config(layout="wide", page_title="Luna's Hybrid Factory")
st.markdown("<style>.stApp { background-color: #0e1117; color: #c9d1d9; } .stTextInput input { background-color: #0d1117 !important; color: #fff; }</style>", unsafe_allow_html=True)

# State Init
if 'character_data' not in st.session_state: st.session_state['character_data'] = defaultdict(list)
if 'batch_candidates' not in st.session_state: st.session_state['batch_candidates'] = [] 
if 'debug_logs' not in st.session_state: st.session_state['debug_logs'] = []
if 'available_models' not in st.session_state: st.session_state['available_models'] = []
# [NEW] 存储当前的混合目标
if 'current_mixed_target' not in st.session_state: st.session_state['current_mixed_target'] = {}

# --- Sidebar ---
with st.sidebar:
    st.title("🎛️ Control Panel")
    
    env_key = os.environ.get("GEMINI_API_KEY")
    api_key = env_key if env_key else st.text_input("Gemini API Key", type="password")
    
    st.divider()
    if api_key and not st.session_state['available_models']:
        with st.spinner("Fetching models..."):
            models = fetch_gemini_models(api_key)
            if models: st.session_state['available_models'] = models; st.rerun()
    
    model_list = st.session_state['available_models']
    selected_model = st.selectbox("Model", model_list, index=0) if model_list else st.selectbox("Model", ["Waiting..."], disabled=True)

    st.divider()
    with st.expander("🛠️ Initial Setup", expanded=(len(st.session_state['character_data']) == 0)):
        if PROMPT_CONFIG:
            user_desc = st.text_area("Description", height=100, placeholder="e.g. A silver-haired tsundere...")
            if st.button("🚀 Initialize", disabled=not (api_key and "(" not in selected_model)):
                with st.spinner("Architecting..."):
                    init_data, err = generate_initial_json(api_key, selected_model, PROMPT_CONFIG, user_desc)
                    if err: st.error(err)
                    else:
                        st.session_state['character_data'] = defaultdict(list)
                        count = 0
                        for k, v in init_data.items():
                            ui_key = KEY_MAPPING.get(k)
                            if not ui_key: # Fuzzy
                                for ck in KEY_MAPPING: 
                                    if k in ck: ui_key = KEY_MAPPING[ck]; break
                            if ui_key and v and v != "...":
                                st.session_state['character_data'][ui_key] = [json.dumps(v, ensure_ascii=False)] if isinstance(v, dict) else [str(v)]
                                count += 1
                        st.success(f"Initialized {count} layers!"); time.sleep(1); st.rerun()

    st.divider()
    batch_size = st.slider("Batch Size", 1, 20, 5)
    st.caption("Layer Weights (Probability)")
    weights = {k: st.slider(k.split(":")[0], 0, 100, 15) for k in LAYER_KEYS}
    st.download_button("💾 JSON", json.dumps(st.session_state['character_data'], indent=2, ensure_ascii=False), "luna_data.json")

# --- Main ---
st.title("🐱 Luna's Hybrid Data Factory")
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📚 Memory Bank")
    st.progress(min(sum(len(v) for v in st.session_state['character_data'].values())/1000, 1.0))
    with st.container(height=500): st.json(st.session_state['character_data'], expanded=False)

with col2:
    if not LAYER_KEYS:
        st.error("No Layers Defined. Check Config.")
    else:
        # [逻辑变更] 每一帧都重新计算混合分布 (或者你可以加个按钮来刷新分布)
        # 为了响应左侧滑块，我们需要实时计算
        target_dist = get_mixed_batch_distribution(st.session_state['character_data'], weights, batch_size)
        st.session_state['current_mixed_target'] = target_dist
        
        with st.container(border=True):
            st.markdown(f"### 🎯 Target: Mixed Batch ({batch_size})")
            
            # 显示分布详情
            cols = st.columns(len(target_dist))
            for idx, (layer_name, count) in enumerate(target_dist.items()):
                short_name = layer_name.split(":")[0] # Layer 1
                st.info(f"**{short_name}**: {count} items")
            
            if st.button(f"Generate Mixed Batch", type="primary", use_container_width=True, disabled=not (api_key and "(" not in selected_model)):
                with st.spinner("Generating Mixed Content..."):
                    items, err = generate_mixed_batch_data(
                        target_dist, 
                        st.session_state['character_data'], 
                        api_key, 
                        selected_model, 
                        PROMPT_CONFIG
                    )
                    if err: st.error(err)
                    else: st.session_state['batch_candidates'] = [{'layer': i['layer'], 'content': i['content'], 'selected': True} for i in items]; st.rerun()

        if st.session_state['batch_candidates']:
            st.divider()
            st.markdown("### 🧬 Review Candidates")
            
            # [UI 更新] 显示每个条目的 Layer Tag
            for i, item in enumerate(st.session_state['batch_candidates']):
                c1, c2 = st.columns([0.05, 0.95])
                with c1: item['selected'] = st.checkbox("", item['selected'], key=f"c_{i}")
                with c2:
                    # 显示所属 Layer 的 Badge
                    st.caption(f"🏷️ **{item['layer']}**")
                    item['content'] = st.text_area("Content", item['content'], height=60, key=f"t_{i}", label_visibility="collapsed")
            
            if st.button("Save Selected", type="primary"):
                saved_count = 0
                for item in st.session_state['batch_candidates']:
                    if item['selected']: 
                        st.session_state['character_data'][item['layer']].append(item['content'])
                        saved_count += 1
                st.session_state['batch_candidates'] = []; st.success(f"Saved {saved_count} items!"); time.sleep(0.5); st.rerun()

# Debug
st.divider()
with st.expander("🐞 Debug Logs"):
    if st.button("Clear"): st.session_state['debug_logs'] = []; st.rerun()
    for log in st.session_state['debug_logs']:
        st.caption(f"[{log['time']}] {log['action']}")
        if log.get('error'): st.error(log['error'])
        st.code(log['prompt'], language="text")
        st.code(log['response'], language="json")
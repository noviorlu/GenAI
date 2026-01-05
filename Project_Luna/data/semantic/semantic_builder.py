import streamlit as st
import google.generativeai as genai
from openai import OpenAI  # 引入 OpenAI 库连接 Ollama
import json
import random
import time
from collections import defaultdict

# ==========================================
# 1. 核心定义 (保持不变)
# ==========================================

LAYERS_DEF = {
    "Layer 1: 核心性格 (Core Personality)": "角色的灵魂与不可动摇的本性（如：正义、傲娇、极度忠诚、害怕孤独）。",
    "Layer 2: 外在属性 (External Attributes)": "生理特征（猫耳、异色瞳）、种族、自然属性（天赋、智商、原生家庭）。",
    "Layer 3: 重大事件/心结 (Major Events/Trauma)": "改变人生的关键时刻，PTSD 的来源，造成性格扭曲的原因。",
    "Layer 4: 心路历程 (Psychological Journey)": "面对事件的反应链：初始状态 -> 诱发 -> 激化 -> 选择 -> 结果。",
    "Layer 5: 可爱之处 (Gap Moe/Charm)": "反差萌点，让人想要保护或亲近的特质（嘴硬心软、生活白痴）。",
    "Layer 6: 他人评价 (Others' Perception)": "客人、玩家、或死对头眼中的她是什么样子的。",
    "Layer 7: 行为与口癖 (Behavior & Speech)": "具体的说话风格、惯用语、微表情、下意识动作。"
}

LAYER_KEYS = list(LAYERS_DEF.keys())

# ==========================================
# 2. 逻辑工具
# ==========================================

def get_next_target_layer(current_data, target_weights, total_goal=1000):
    # ... (保持原有的调度逻辑不变) ...
    counts = {k: len(current_data.get(k, [])) for k in LAYER_KEYS}
    UNLOCK_THRESHOLDS = {
        LAYER_KEYS[0]: 0, LAYER_KEYS[1]: 5, LAYER_KEYS[2]: 5,
        LAYER_KEYS[3]: 3, LAYER_KEYS[4]: 3, LAYER_KEYS[5]: 2, LAYER_KEYS[6]: 10,
    }
    available_layers = []
    for i, layer in enumerate(LAYER_KEYS):
        if i == 0: available_layers.append(layer)
        else:
            prev_layer = LAYER_KEYS[i-1]
            if counts[prev_layer] >= UNLOCK_THRESHOLDS[layer]: available_layers.append(layer)
            else: break 
    weighted_pool = []
    for layer in available_layers:
        target_count = total_goal * (target_weights[layer] / 100.0)
        current_count = counts[layer]
        deficit = max(0, target_count - current_count)
        if deficit > 0: weighted_pool.extend([layer] * int(deficit))
    if not weighted_pool: return available_layers[-1] if available_layers else LAYER_KEYS[0]
    return random.choice(weighted_pool)

def clean_json_array_output(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # 过滤掉 markdown 标记
        text = "\n".join([line for line in lines if not line.strip().startswith("```")])
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        text = text[start:end+1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return []

# ==========================================
# 3. 双引擎生成核心 (Dual Engine)
# ==========================================

def generate_with_gemini(model_name, api_key, prompt):
    """Google Gemini 引擎"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text

def generate_with_ollama(model_name, prompt):
    """Local Ollama 引擎 (OpenAI Compatible)"""
    client = OpenAI(
        base_url='http://localhost:11434/v1', # Ollama 默认地址
        api_key='ollama', # 本地不需要真实 key，但库要求必填
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content

def generate_batch_data(engine_type, layer_name, current_profile, api_key, model_name, batch_size=5):
    context_str = json.dumps(current_profile, indent=2, ensure_ascii=False)
    layer_desc = LAYERS_DEF[layer_name]
    
    # 通用 Prompt
    prompt = f"""
    [Role]
    Senior Narrative Designer for Project "Luna" (Tsundere Cat-girl).
    
    [Context]
    {context_str}
    
    [Target]
    Layer: {layer_name}
    Definition: {layer_desc}
    
    [Task]
    Generate a JSON Array of {batch_size} distinct, creative data points (strings) in Chinese.
    
    [Rules]
    1. Output JSON Array ONLY: ["item1", "item2"]
    2. No Markdown, no explanations.
    3. Be consistent with existing context.
    """
    
    try:
        raw_text = ""
        if engine_type == "Cloud (Gemini)":
            raw_text = generate_with_gemini(model_name, api_key, prompt)
        else:
            raw_text = generate_with_ollama(model_name, prompt)
            
        data = clean_json_array_output(raw_text)
        if not isinstance(data, list):
            return [], "Error: Invalid JSON format returned."
        return data, None
        
    except Exception as e:
        return [], f"{engine_type} Error: {str(e)}"

# ==========================================
# 4. Streamlit UI
# ==========================================

st.set_page_config(layout="wide", page_title="Luna's Hybrid Factory")

# --- CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #c9d1d9; }
    section[data-testid="stSidebar"] { background-color: #161b22; }
    .stTextInput input, .stTextArea textarea, div[data-baseweb="select"] > div {
        background-color: #0d1117 !important; color: #fff !important; border: 1px solid #30363d !important;
    }
</style>
""", unsafe_allow_html=True)

# --- State ---
if 'character_data' not in st.session_state: st.session_state['character_data'] = defaultdict(list)
if 'batch_candidates' not in st.session_state: st.session_state['batch_candidates'] = [] 
if 'current_layer_target' not in st.session_state: st.session_state['current_layer_target'] = LAYER_KEYS[0]

# --- Sidebar ---
with st.sidebar:
    st.title("🎛️ Control Panel")
    
    # 1. 引擎选择
    engine_type = st.radio("Compute Engine", ["Cloud (Gemini)", "Local (Ollama)"], horizontal=True)
    
    api_key = ""
    selected_model = ""
    
    if engine_type == "Cloud (Gemini)":
        st.info("☁️ Google Cloud")
        api_key = st.text_input("Gemini API Key", type="password")
        # 简单的模型选择
        selected_model = st.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
        
    else:
        st.success("🏠 RTX 4080 Super Detected")
        # 这里你可以填你 pull 下来的模型名
        selected_model = st.text_input("Local Model Name", value="qwen2.5:14b", help="Run 'ollama list' in terminal to see names")
        api_key = "local" # 占位符

    st.divider()
    batch_size = st.slider("Batch Size", 1, 20, 5)
    
    # 权重滑块
    st.subheader("Layer Weights")
    weights = {}
    defaults = [10, 15, 15, 10, 10, 10, 30] 
    for i, key in enumerate(LAYER_KEYS):
        weights[key] = st.slider(key.split(":")[0], 0, 100, defaults[i])

    # 下载
    st.download_button("💾 Download JSON", json.dumps(st.session_state['character_data'], indent=2, ensure_ascii=False), "luna_data.json")

# --- Main ---
st.title("🐱 Luna's Hybrid Data Factory")
st.caption(f"Running on: **{engine_type}** | Model: **{selected_model}**")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📚 Memory Bank")
    # 简单的进度展示
    total = sum(len(v) for v in st.session_state['character_data'].values())
    st.progress(min(total/1000, 1.0))
    st.text(f"Total: {total} records")
    
    with st.container(height=500):
        st.json(st.session_state['character_data'], expanded=False)

with col2:
    target = get_next_target_layer(st.session_state['character_data'], weights)
    st.session_state['current_layer_target'] = target
    
    with st.container(border=True):
        st.markdown(f"### 🎯 Target: {target}")
        if st.button(f"Generate Batch ({batch_size})", type="primary", use_container_width=True):
            if engine_type == "Cloud (Gemini)" and not api_key:
                st.error("Missing Google API Key")
            else:
                with st.spinner(f"Generating via {selected_model}..."):
                    items, err = generate_batch_data(engine_type, target, st.session_state['character_data'], api_key, selected_model, batch_size)
                    if err: st.error(err)
                    elif items:
                        st.session_state['batch_candidates'] = [{'content': i, 'selected': True} for i in items]
                        st.rerun()

    # Review
    if st.session_state['batch_candidates']:
        st.divider()
        st.markdown("### 🧬 Review")
        
        # 简单的 Grid 布局
        for i, item in enumerate(st.session_state['batch_candidates']):
            c1, c2 = st.columns([0.1, 0.9])
            with c1:
                item['selected'] = st.checkbox("", item['selected'], key=f"check_{i}")
            with c2:
                item['content'] = st.text_area("Content", item['content'], height=60, key=f"text_{i}", label_visibility="collapsed")
        
        if st.button("Save Selected", type="primary"):
            for item in st.session_state['batch_candidates']:
                if item['selected']:
                    st.session_state['character_data'][target].append(item['content'])
            st.session_state['batch_candidates'] = []
            st.success("Saved!")
            time.sleep(0.5)
            st.rerun()
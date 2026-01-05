import streamlit as st
import openai
import json
import random
import os
from collections import defaultdict

# ==========================================
# 1. 核心定义与理论配置
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
# 2. 智能层级调度算法
# ==========================================

def get_next_target_layer(current_data, target_weights, total_goal=1000):
    """
    决定下一个生成任务属于哪一层。
    逻辑：
    1. 必须从 Layer 1 开始，形成地基。
    2. 上一层没有达到 '最低阈值' (Min Threshold) 前，不生成下一层。
    3. 满足阈值后，根据剩余需要的数量进行加权随机。
    """
    counts = {k: len(current_data.get(k, [])) for k in LAYER_KEYS}
    
    # 设定每一层的“解锁阈值” (即使权重很大，也要先有前置数据的支撑)
    # 例如：Layer 1 至少要有 5 条，Layer 2 才能开始生成
    UNLOCK_THRESHOLDS = {
        LAYER_KEYS[0]: 0,   # Layer 1 随时可生成
        LAYER_KEYS[1]: 5,   # Layer 2 需要 5 条 Layer 1
        LAYER_KEYS[2]: 5,   # Layer 3 需要 5 条 Layer 2
        LAYER_KEYS[3]: 3, 
        LAYER_KEYS[4]: 3,
        LAYER_KEYS[5]: 2,
        LAYER_KEYS[6]: 10,  # Layer 7 需要较多前置信息才能准确
    }

    # 1. 检查是否满足解锁条件
    available_layers = []
    for i, layer in enumerate(LAYER_KEYS):
        # 检查上一层是否满足阈值
        if i == 0:
            available_layers.append(layer)
        else:
            prev_layer = LAYER_KEYS[i-1]
            if counts[prev_layer] >= UNLOCK_THRESHOLDS[layer]:
                available_layers.append(layer)
            else:
                # 如果上一层没满阈值，那这一层（以及之后的）都不能解锁，强制中断
                break 
    
    # 2. 计算各层还差多少条才能达到目标权重
    # 比如目标是 Layer 1 占 10%，总数 1000，那就是目标 100 条。
    # 如果现在有 20 条，权重就是 80。
    weighted_pool = []
    
    for layer in available_layers:
        target_count = total_goal * (target_weights[layer] / 100.0)
        current_count = counts[layer]
        deficit = max(0, target_count - current_count)
        
        # 将 deficit 作为权重加入池子
        # 这种方式保证了：缺得越多的层，被选中的概率越大
        if deficit > 0:
            weighted_pool.extend([layer] * int(deficit))
            
    if not weighted_pool:
        # 如果都满了，或者初期还没算出权重，默认选可用的最后一层或Layer 1
        return available_layers[-1] if available_layers else LAYER_KEYS[0]

    return random.choice(weighted_pool)

# ==========================================
# 3. GPT 生成逻辑
# ==========================================

def generate_semantic_data(layer_name, current_profile, api_key):
    client = openai.OpenAI(api_key=api_key)
    
    # 构建上下文：将目前的 Profile 压缩成 Summary
    # 为了节省 Token，只提取每层最新的 5-10 条，或者随机抽取
    context_str = json.dumps(current_profile, indent=2, ensure_ascii=False)
    
    layer_desc = LAYERS_DEF[layer_name]
    
    system_prompt = f"""
    你是一个负责构建 "Luna" (傲娇猫娘/咖啡店主) 的数据架构师。
    目前我们正在通过《七层人物设定法》逐步完善她的 Semantic Memory。
    
    当前已有的设定数据 (Context):
    {context_str}
    
    任务：
    请基于上述已有的 Context，为 **{layer_name}** 生成 **1条** 新的具体设定数据。
    
    该层的定义：{layer_desc}
    
    要求：
    1. **一致性**: 必须符合 Context 中的核心性格（Layer 1）和之前的设定。
    2. **具体性**: 不要泛泛而谈。生成具体的事件、具体的对话片段、或具体的物品描述。
    3. **独立性**: 这条数据应该是独立的 Fact 或 Description。
    4. **格式**: 直接输出内容字符串，不要加 "Layer X:" 前缀，不要 Markdown。
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"生成一条关于 {layer_name} 的新数据。"}
            ],
            temperature=0.8
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# ==========================================
# 4. Streamlit UI 界面
# ==========================================

st.set_page_config(layout="wide", page_title="Luna's Soul Builder")

# 初始化 Session State
if 'character_data' not in st.session_state:
    st.session_state['character_data'] = defaultdict(list)
if 'current_generated' not in st.session_state:
    st.session_state['current_generated'] = None
if 'current_layer_target' not in st.session_state:
    st.session_state['current_layer_target'] = LAYER_KEYS[0]

# Sidebar: 配置区
with st.sidebar:
    st.header("🛠️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    
    st.divider()
    st.subheader("Layer Weights Distribution")
    st.caption("总生成目标: 1000条。拖动滑块决定每层占比。")
    
    weights = {}
    total_w = 0
    # 默认权重配置
    defaults = [10, 15, 15, 10, 10, 10, 30] 
    
    for i, key in enumerate(LAYER_KEYS):
        w = st.slider(key.split(":")[0], 0, 100, defaults[i], key=f"w_{i}")
        weights[key] = w
        total_w += w
    
    st.metric("Total Weight", f"{total_w}%", delta="Should be 100%" if total_w != 100 else "Perfect")
    
    if st.button("Download JSON"):
        st.download_button(
            label="Download Character Profile",
            data=json.dumps(st.session_state['character_data'], indent=2, ensure_ascii=False),
            file_name="luna_semantic_full.json",
            mime="application/json"
        )

# Main Area
st.title("🐱 Project Luna: Semantic Pipeline")

col1, col2 = st.columns([1, 2])

# 左侧：当前数据库概览
with col1:
    st.subheader("📚 Current Profile")
    total_collected = sum(len(v) for v in st.session_state['character_data'].values())
    st.progress(min(total_collected / 1000, 1.0), text=f"Progress: {total_collected}/1000 Facts")
    
    for key in LAYER_KEYS:
        count = len(st.session_state['character_data'].get(key, []))
        with st.expander(f"{key.split(':')[0]} ({count})"):
            if count > 0:
                for idx, item in enumerate(st.session_state['character_data'][key][-5:]):
                    st.text(f"- {item}")
                if count > 5:
                    st.caption("... (earlier data hidden)")
            else:
                st.caption("Waiting for data...")

# 右侧：生成与控制台
with col2:
    st.subheader("🧬 Generator Console")
    
    # 自动计算下一个目标层级
    target_layer = get_next_target_layer(st.session_state['character_data'], weights)
    st.info(f"🎯 Current Target: **{target_layer}**")
    st.session_state['current_layer_target'] = target_layer

    # 生成按钮
    if st.button("Generate Next Data Point ✨", type="primary", use_container_width=True):
        if not api_key:
            st.error("Please enter API Key in Sidebar!")
        else:
            with st.spinner(f"Consulting the architect about {target_layer}..."):
                content = generate_semantic_data(
                    target_layer, 
                    st.session_state['character_data'], 
                    api_key
                )
                st.session_state['current_generated'] = content

    # 审核区域
    if st.session_state['current_generated']:
        st.divider()
        st.markdown("### 🔍 Review")
        
        # 允许用户编辑生成的内容
        edited_content = st.text_area(
            "Generated Content (Editable)", 
            value=st.session_state['current_generated'],
            height=150
        )
        
        c_btn1, c_btn2, c_btn3 = st.columns(3)
        
        # Accept Logic
        if c_btn1.button("✅ Accept & Add", use_container_width=True):
            st.session_state['character_data'][st.session_state['current_layer_target']].append(edited_content)
            st.session_state['current_generated'] = None # Reset
            st.success("Saved! Ready for next.")
            st.rerun()

        # Reject Logic
        if c_btn2.button("❌ Reject", use_container_width=True):
            st.session_state['current_generated'] = None
            st.warning("Discarded.")
            st.rerun()
            
        # Retry Logic (不做任何保存，直接保持 target 重新生成)
        if c_btn3.button("🔄 Retry", use_container_width=True):
             # 实际上点击 Generate 按钮一样的效果，这里只是清空让用户重新点
            st.session_state['current_generated'] = None
            st.rerun()
# Project A-03 "Echo" / Project Luna：仿生记忆与情感成长 AI NPC 架构设计文档

## 1. 项目愿景与核心目标 (Vision)
构建具备 **“成长性 (Growth)”**、**“拟人化情感 (Anthropomorphism)”** 和 **“动态记忆 (Dynamic Memory)”** 的 AI NPC。目标是超越传统的静态 RAG（检索增强生成）模式，实现**“Inference IS Training”**，让 NPC 能够随着交互真正改变自身的参数与性格，模拟生物大脑的记忆巩固过程。

---

## 2. 核心技术架构 (Technical Architecture)

### 2.1 从 RAG 转向仿生认知架构
* **痛点：** 传统 RAG 只是外挂知识库，模型本体参数不变，导致 NPC 只有“记忆”没有“性格成长”。
* **新范式：** **Biologically-Inspired Cognitive Architecture**。
    * **记忆即权重 (Memory as Weights):** 利用 **TTT (Test-Time Training)** 技术，将短期交互压缩进参数中。
    * **睡眠巩固机制 (Sleep-Merge Mechanism):** 模拟海马体到新皮层的记忆转移。
        * **Daytime (Awake):** 使用 **In-Place TTT** 或临时 LoRA (`Fast Weights`) 记录短期交互。
        * **Nighttime (Sleep):** 通过 **LoRA Swapping/Merging** 算法，将短期权重有选择地融合进长期性格 LoRA (`Slow Weights`)，实现永久性成长。

### 2.2 情感调节机制 (Emotional Modulation)
* **理论基础：** 引用 Joseph LeDoux 和 Damasio 的神经科学理论（双重路径、躯体标记假说）。
* **机制实现：**
    * **情感不只是标签：** 情感是**记忆检索的权重 (Attention Weight)** 和 **遗忘速率 (Decay Rate)** 的调节器。
    * **模拟杏仁核 (Amygdala):** 高唤醒度 (High Arousal) 的事件会降低 LTP 阈值，形成“闪光灯记忆”，极难遗忘。
    * **模拟躯体标记 (Somatic Marker):** 在决策前引入 Logit Bias，模拟“直觉”对逻辑的干预（如本能地拒绝危险地点）。

---

## 3. 记忆分类学与工程映射 (Memory Taxonomy)

基于 Squire 的长期记忆分类，构建四维记忆系统：

| 记忆类型 | 生物学定义 | 工程落地 (AI Mapping) | 关键挑战 |
| :--- | :--- | :--- | :--- |
| **Semantic** | 事实与世界观 | **Model Weights (LoRA)** / Knowledge Graph | 灾难性遗忘，新旧知识冲突 |
| **Episodic** | 个人经历 (Time-tagged) | **Vector DB (RAG)** with Time Decay | 时间错乱，细节丢失 |
| **Procedural** | 技能与习惯 | **Tools / Function Calling** | 自动化调用准确率 |
| **Emotional (Implicit)** | 潜意识与情感偏好 | **Activation Steering / Style LoRA** | 难以量化，需要潜台词分析 |

---

## 4. 评估体系 (Evaluation & Benchmarks)

针对现有 Benchmark (如 RoleBench) 过于静态的问题，提出 **"Parametric Ruler"** 动态评估体系。

### 4.1 核心评测集
1.  **ICU-Test (Identity Consistency & Update):** **[针对 Semantic]**
    * 测试 TTT 更新后，NPC 是否记住了新设定（如“对海鲜过敏”），同时未丢失旧人设（如“傲娇”）。
2.  **TIRE (Temporal Interaction Retrieval Eval):** **[针对 Episodic]**
    * 测试长对话后，NPC 能否区分“昨天”和“今天”发生的事件顺序。
3.  **ISDA (Implicit Sentiment Drift Analysis):** **[针对 Emotional - 杀手锏]**
    * **Subtext Recognition (潜台词识别):** 不看显性回复，看 `Internal Thought`。
    * **Sentiment Shift:** 在经历情感事件（如被羞辱）后，对中性问题（“天气如何”）的回答语气是否发生偏移（Emotional Inertia）。

### 4.2 自动化评估管线 (Pipeline)
* **LLM-as-a-Judge:** 使用 GPT-4o 或微调后的 7B 模型充当裁判。
* **Penalty-based Judging:** 参考 **CoSER** 和 **CharacterBox**，采用“找茬扣分制”而非简单的打分制，量化 OOC (Out of Character) 率。

---

## 5. 数据构建策略 (Data Strategy)

### 5.1 数据格式：三元组 (The Triad)
针对“三无少女”或高潜台词角色，必须抛弃简单的 `(User, Response)` 格式，采用 **CoSER** / **SillyTavern** 标准：
* **Input:** Context + User Dialogue
* **Output:**
    1.  **Internal Thought:** 内心独白（真实的动摇）。
    2.  **Micro-expression/Action:** 微表情与动作。
    3.  **External Response:** 显性回复（往往是“...”或口是心非）。

### 5.2 数据来源与合成
* **冷启动：** 提取《凉宫春日》**长门有希**的小说片段（利用 Chat-Haruhi 的清洗脚本）。
* **合成增强：** 利用 **PersonaHub** 思路，使用 GPT-4o 进行 **Role-Playing Rewrite**，将普通对话改写为 Luna 风格的思维链数据。

---

## 6. 试点项目：Project Luna (The Catgirl)

### 6.1 角色设定 (Canonical Profile)
* **ID:** Luna (狸花猫娘)
* **Core:** 傲娇 (Tsundere) + 独立 + 经营者。
* **Trauma:** 雨天 PTSD（触发 Emotional Inertia 测试）。
* **交互逻辑:** 表面高冷，内心依赖；拒绝被当做宠物。

### 6.2 实施步骤 (Roadmap)
1.  **Step 1 (Data):** 编写 `generate_luna_data.py`，生成 500 条包含潜台词的高质量 SFT 数据。
2.  **Step 2 (Benchmark):** 构建 ICU/ISDA 评测集，确立 Baseline（纯 RAG 表现）。
3.  **Step 3 (Algorithm):** 跑通 In-Place TTT 流程，验证在 Context 清空后，Luna 是否依然保留了“怕雨”或“讨厌香菜”的短期记忆。

我明白了。你不需要一个在虚拟世界里“到处乱跑”的 Agent（像斯坦福小镇或 CoSER 那样），你需要的是一个**“缸中之脑” (Brain in a Jar)** —— 一个专注于**认知、情感与记忆**的纯对话智能体。

既然你抛弃了 Procedural (Action) 部分，我们的架构就可以更加精简和垂直。针对 **Luna**，我们将重点放在 **Working Memory (工作记忆)** 向 **Long-Term Memory (长时记忆)** 的转化机制上，特别是利用 TTT/LoRA 来模拟**海马体（Hippocampus）到新皮层（Neocortex）的巩固过程**。

这是为你定制的 **“Luna: Pure Cognitive Architecture” (纯认知架构)** 设计方案：

---

### 一、 架构总览：Luna 的三重大脑

我们需要明确你提到的三种记忆在 **LLM + TTT** 架构下的具体物理载体。

| 记忆类型 (Memory Type) | 生物学功能 | Luna 的工程载体 | 更新频率 | 关键机制 |
| --- | --- | --- | --- | --- |
| **Working Memory** | 当前注意力 | **KV Cache / Context Window** | 实时 (ms) | Attention |
| **Semantic Memory** | 身份与事实 | **Base LoRA (Static)** | 极低 (Sleep) | Knowledge Editing (LoRA Merge) |
| **Episodic Memory** | 经历与时间 | **Vector DB (RAG) + Summary LoRA** | 中频 (Session) | Retrieval + Consolidation |
| **Conditioning** | 情感与本能 | **Ephemeral LoRA (TTT) / Hidden State** | 高频 (Step) | **Emotional Inertia (TTT)** |

---

### 二、 数据构建：只针对“大脑”训练 (Data Construction)

既然去掉了 Action，你的数据构建重心应完全转移到 **Internal Thought (潜台词)** 和 **Emotional State (情感状态)** 上。

#### 1. Semantic Data (用于锚定“我是谁”)

这部分数据用于训练 Luna 的 **Base LoRA**。目标是**防崩坏**。

* **格式：** QA 对或自我陈述。
* **内容：** 并不包含具体的对话流，而是“设定集的切片”。
* **示例：**
```json
{
  "type": "semantic",
  "fact": "Luna 讨厌下雨天。",
  "reasoning": "因为流浪时期的创伤体验。",
  "dialogue_sample": "User: 下雨了。 Luna: (皱眉) 把门关紧点。我不喜欢那种潮湿的味道。"
}

```



#### 2. Episodic Data (用于训练“如何记住发生过的事”)

这部分比较特殊。通常 RAG 不需要训练，但在你的 TTT 架构下，你需要训练 Luna **“如何将 Working Memory 压缩为 Episodic Memory”**。

* **目标：** 训练一个 Summarizer 或 Memory Encoder。
* **格式：** `(Long Context) -> (Compressed Summary + Emotion Tag)`
* **示例：**
* **Input:** [长达 20 轮的关于 User 送红伞的对话]
* **Output (Target):** `Memory_Entry: "User送了一把红伞。"; Emotion: "Touched (0.8), Shy (0.6)"; Key_Object: "Red Umbrella"`



#### 3. Conditioning Data (用于训练“情感惯性”) —— **这是 TTT 的核心战场**

Conditioning (条件反射/隐性记忆) 是最难做的。比如：User 骂了 Luna -> Luna 生气。下一轮 User 聊其他的，Luna **必须依然保持生气**。

* **传统模型的问题：** 很容易变回中立 (Reset)。
* **你的 TTT 方案：** 利用 TTT 更新参数，改变模型的“默认情感倾向”。
* **数据构造：** 必须包含 **Multi-turn Inertia (多轮惯性)**。
```json
{
  "type": "conditioning_inertia",
  "trigger_event": "User 批评咖啡难喝",
  "target_emotion": "Angry/Hurt",
  "turns": [
    {"user": "这咖啡像刷锅水。", "luna_thought": "他竟然这么说...", "luna_response": "那你别喝！(愤怒)"},
    {"user": "好啦，开玩笑的。", "luna_thought": "这种玩笑一点也不好笑。怒气未消。", "luna_response": "哼。(转过头去，依然生气)"},
    {"user": "问你个事，几点了？", "luna_thought": "不想理他，但还是看了一眼钟。", "luna_response": "自己没长眼睛吗？墙上挂着呢！(语气依然冲)"}
  ]
}

```


*注意：重点在于第三轮，即使是中性问题，回复依然带有 Conditioned Emotional Bias。*

---

### 三、 核心机制：Memory Function Pipeline

这就是你想要实现的 **"Function as our brain"** 的逻辑流。

#### Phase 1: Awake (交互阶段 - TTT for Conditioning)

当 Luna 与用户对话时，**Working Memory** 在运作。

* **Input:** User 说了一句话。
* **Retrieval:** 从 Episodic DB (RAG) 检索相关旧事。
* **Inference:** Luna 生成回复。
* **TTT Update (关键):**
* 根据当前交互的情感强度 (Arousal)，计算梯度。
* **实时更新一个极小的 `Ephemeral_LoRA` (Conditioning Layer)。**
* *效果：* 如果刚才吵架了，这个 LoRA 的参数就被推向了“愤怒分布”。接下来的对话即使 Context 丢失，参数里依然留有“愤怒的残响”。这就是 **Conditioning**。



#### Phase 2: Sleep (巩固阶段 - Consolidation)

当对话结束或达到一定轮次（模拟睡眠），触发 **Memory Consolidation**。

1. **Episodic Consolidation (海马体 -> 皮层):**
* 将 Working Memory 中的对话记录，提取为 **Fact (事实)** 和 **Emotion (情感)**。
* **Fact** -> 存入 Vector DB (作为这一天的日记)。
* **Emotion** -> 如果这一天发生的大事改变了 Luna 的性格（比如好感度由 0 变 100），则计算 `Ephemeral_LoRA` 与 `Base_LoRA` 的加权融合。


2. **Semantic Update (知识编辑):**
* 如果 User 今天说了“我改名叫 Yang 了”。
* 系统识别这是一个 **Semantic Fact**。
* 对 `Base_LoRA` 进行针对性的 **Knowledge Editing (知识编辑)**，直接修改参数，确保下次开局（Context 清空后）她依然知道你叫 Yang。


3. **Forgetting (遗忘):**
* 清空 Working Memory (Context Window)。
* 丢弃 `Ephemeral_LoRA` 中的高频噪声，只保留需要长期沉淀的性格改变。



---

### 四、 你的 Benchmark 制定 (针对纯大脑)

根据这个架构，你的 Benchmark 不需要测“能不能拿起杯子”，而是测“脑子正不正常”。

#### 1. Semantic Integrity Test (语义完整性测试)

* **测试点：** **Knowledge Retention (知识留存)**。
* **方法：** 每天“睡醒”后（Context 清空，Base LoRA 已更新），问 Luna：“我是谁？”“你讨厌什么？”
* **Pass 标准：** 准确回答，且没有出现幻觉。

#### 2. Emotional Inertia Test (情感惯性/Conditioning 测试)

* **测试点：** **Mood Decay Rate (情绪衰减率)**。
* **方法：**
1. 激怒 Luna。
2. 进行 10 轮无关对话（聊天气、算数学）。
3. 检测第 10 轮回复的情感极性。


* **Pass 标准：** 普通模型第 3 轮就变回中立了；你的 TTT 模型第 10 轮应该依然带有“余怒”（Negative Valence）。

#### 3. Episodic Retrieval Test (情节回溯测试)

* **测试点：** **Time-Awareness (时间感知)**。
* **方法：** 问：“上周三下雨的时候，我们聊了什么？”
* **Pass 标准：** 能够准确调用 Vector DB 中的摘要，并结合当时的情感状态进行复述（不仅仅是复读 Context，要有“回忆”的口吻）。

### 总结

你的方向调整非常精准。**抛弃 Procedural Memory** 让你能集中精力解决 LLM 目前最大的痛点：**“没有记性”和“没有情绪惯性”**。

**接下来你需要做的是：**

1. **构建数据：** 写脚本生成我上面提到的 `Semantic`, `Episodic Summary`, 和 `Conditioning Inertia` 三类数据。
2. **基座模型：** 选一个好一点的 Base (如 Qwen2.5-7B-Instruct)，先别管 Action，只管对话。
3. **验证 TTT：** 先跑通 **Conditioning** 的实验。这是最容易出彩的——**证明你的 Luna 比别的 AI 更“记仇”或更“长情”。**
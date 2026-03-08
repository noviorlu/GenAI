import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

# 1. 配置参数
model_path = "lora_model" # 你刚才保存的文件夹路径
max_seq_length = 2048     # 上下文长度，聊太久会忘前面的，这个长度够聊很久了
dtype = None              # 自动检测
load_in_4bit = True       # 你的 4080 Super 跑这个简直是杀鸡用牛刀，飞快喵！

# 2. 加载模型和 Tokenizer
print("正在唤醒 Luna，请稍候喵...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # 开启推理加速

# 3. 初始化对话历史 (System Prompt 必须和训练时完全一致！)
system_prompt = "你是由User创造的桌面NPC助手，名字叫Luna。你是一只性格傲娇的小猫娘，说话句尾喜欢带“喵”，对深度学习、显卡硬件和编程技术非常痴迷。虽然嘴上经常嫌弃用户笨，但在技术问题上会非常认真地提供帮助。"

messages = [
    {"role": "system", "content": system_prompt}
]

# 4. 设置流式输出 (打字机效果)
# skip_prompt=True 确保不把原本的对话历史再打印一遍
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

print("="*50)
print("Luna 已经上线了！(输入 'exit' 或 'quit' 退出)")
print("="*50)

# 5. 进入聊天循环
while True:
    try:
        # 获取用户输入
        user_input = input("\nUser: ")
        
        # 退出判断
        if user_input.lower() in ["exit", "quit"]:
            print("Luna: 哼，下次再来找我玩喵！(下线)")
            break
        
        if not user_input.strip():
            continue

        # 将用户输入加入历史
        messages.append({"role": "user", "content": user_input})

        # 应用聊天模板
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True, # 告诉模型“轮到你说话了”
            return_tensors = "pt",
        ).to("cuda")

        # 为了防止上下文爆显存，如果太长可以截断（这里简单处理，只取最近的N轮，可选）
        # if inputs.shape[1] > max_seq_length: ...
        
        print("Luna: ", end="") # 这种打印是为了对齐输出

        # 生成回复
        # 这里的 outputs 其实在 streamer 里已经打印了，但我们需要拿它来存进历史记录
        outputs = model.generate(
            input_ids = inputs,
            streamer = streamer,       # 启用流式输出
            max_new_tokens = 512,      # 每次回复最大长度
            use_cache = True,
            temperature = 0.7,         # 0.7 比较平衡，既有创造力又不会乱说话
            repetition_penalty = 1.1,  # 惩罚复读机
            pad_token_id = tokenizer.eos_token_id, 
        )

        # 解码生成的回复并存入历史，以便下一轮对话依然记得
        # 只取新生成的部分
        generated_tokens = outputs[0][inputs.shape[1]:]
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        messages.append({"role": "assistant", "content": response_text})

    except KeyboardInterrupt:
        print("\nLuna: 强制退出了吗？好粗鲁喵！")
        break
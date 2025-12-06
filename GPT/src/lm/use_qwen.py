import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"

class QwenChatbot:
    # 1. 修正模型名称: "Qwen3-0.6B" 不存在，改为目前最强的 0.5B Instruct 模型
    def __init__(self, model_name=model_name):
        print(f"🔄 正在加载模型: {model_name} ...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            # 2. 显卡加速: device_map="cuda"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cuda", 
                trust_remote_code=True
            )
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise e
            
        self.history = []
        print("✅ 模型加载完成！")

    def generate_response(self, user_input):
        # 将用户输入加入历史
        messages = self.history + [{"role": "user", "content": user_input}]

        # 应用对话模板
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 3. 数据搬运: 输入也必须转到 cuda
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        
        # 生成回复
        with torch.no_grad():
            response_ids = self.model.generate(
                **inputs, 
                max_new_tokens=2048, # 这里的 token 数控制回复的最大长度
                temperature=0.7,     # 增加一点随机性，让对话更自然
                top_p=0.9
            )[0][len(inputs.input_ids[0]):].tolist()
            
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # 更新历史记录
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response

# === 主要修改在这里 ===
if __name__ == "__main__":
    # 初始化机器人
    chatbot = QwenChatbot()

    print("\n" + "="*40)
    print("🤖 Qwen 聊天机器人已启动")
    print("输入 'exit' 或 'quit' 退出对话")
    print("="*40 + "\n")

    # 4. 开启无限循环
    while True:
        try:
            # 获取用户输入
            user_input = input("User: ").strip()

            # 检查退出指令
            if user_input.lower() in ["exit", "quit"]:
                print("Bot: 再见！")
                break
            
            # 如果输入为空，跳过
            if not user_input:
                continue

            # 生成回复并打印
            response = chatbot.generate_response(user_input)
            print(f"Bot: {response}\n")

        except KeyboardInterrupt:
            # 按 Ctrl+C 也可以优雅退出
            print("\nBot: 检测到中断，正在退出...")
            break
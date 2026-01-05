import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. Load Data
try:
    history = torch.load("luna_gradient_trajectory.pt")
    print(f"Loaded trajectory with {len(history)} steps.")
except FileNotFoundError:
    print("Error: Could not find 'luna_gradient_trajectory.pt'. Please run the training script first.")
    exit()

# 2. Aggregation Strategy (聚合策略)
# 我们不仅仅看某一步，我们看整个 "刻写过程" 中，哪里积累的梯度能量最大。
# 方法：对所有 Step 的梯度求和 (Sum) 或 取平均 (Mean)。
# 这里我们用 "Sum" 来代表总共花费了多少能量去修改参数。

# 初始化聚合字典
aggregated_grads = {}
modules = set()
layers = set()

for step_data in history:
    for key, norm in step_data.items():
        # key format: "L{layer}_{module}" e.g., "L0_q_proj"
        if key not in aggregated_grads:
            aggregated_grads[key] = 0.0
        aggregated_grads[key] += norm
        
        # Parse for axis labels
        parts = key.split('_', 1) # Split at first underscore
        layer_str = parts[0] # "L0"
        module_str = parts[1] # "q_proj"
        
        layers.add(int(layer_str[1:])) # extract number
        modules.add(module_str)

# 3. Prepare Matrix for Heatmap
sorted_layers = sorted(list(layers))
sorted_modules = sorted(list(modules)) # e.g. ['down_proj', 'gate_proj', ...]

# 这里的排序建议：按照 Transformer 内部数据流顺序手动排序，图表会更符合直觉
ordered_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
# 过滤掉不在数据里的模块
sorted_modules = [m for m in ordered_modules if m in sorted_modules]

data_matrix = np.zeros((len(sorted_modules), len(sorted_layers)))

for i, mod in enumerate(sorted_modules):
    for j, layer in enumerate(sorted_layers):
        key = f"L{layer}_{mod}"
        val = aggregated_grads.get(key, 0.0)
        data_matrix[i, j] = val


# 使用对数标度 (Log Scale) 通常能看得更清楚，因为某些层的梯度可能比其他层大几个数量级
# 如果数据差异没那么大，可以去掉 np.log1p
plot_data = data_matrix 
# plot_data = np.log1p(data_matrix) # Uncomment if contrast is too high
# --- 4. Plotting with Annotations (PhD Style) ---
plt.figure(figsize=(16, 8), dpi=150) # 把图变得更宽一些，给数字留空间
sns.set_theme(style="white")

# 格式说明：
# fmt=".2f" 表示保留两位小数。
# annot_kws={"size": 8} 控制数字字体大小，如果格子太密，可以调小这个值(如 6)。
ax = sns.heatmap(
    plot_data,
    xticklabels=sorted_layers,
    yticklabels=sorted_modules,
    cmap="rocket",       # 推荐配色
    annot=True,          # <--- 开启数字标注
    fmt=".2f",           # <--- 数字格式：保留2位小数
    annot_kws={"size": 9}, # <--- 数字字体大小
    cbar_kws={'label': 'Cumulative Gradient Norm (Energy)'},
    linewidths=0.5,      # 格子间距
    square=False
)

plt.title("The Anatomy of a Memory: Gradient Heatmap (with Values)", fontsize=16, pad=20)
plt.xlabel("Layer Index (Depth)", fontsize=14)
plt.ylabel("Module Type", fontsize=14)

# 调整X轴标签角度，防止重叠
plt.xticks(rotation=0) 
plt.yticks(rotation=0)

plt.tight_layout()

# Save
plt.savefig("luna_memory_heatmap_20.png")
print("🖼️ Annotated Heatmap generated: luna_memory_heatmap.png")
plt.show()
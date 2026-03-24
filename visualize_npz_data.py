import numpy as np
import os
import matplotlib.pyplot as plt

def visualize_npz(npz_file_path):
    print(f"正在读取文件: {npz_file_path}")
    data = np.load(npz_file_path)
    keys = list(data.keys())
    
    print(f"里面包含 {len(keys)} 个天气变量的统计信息。")
    print(f"前 5 个变量名称: {keys[:5]}\n")
    
    # 我们随便挑一个变量看看它的形状
    sample_var = keys[0]
    sample_data = data[sample_var]
    
    print(f"变量 '{sample_var}' 的数据形状是: {sample_data.shape}")
    print(f"变量 '{sample_var}' 的具体数据内容是:")
    print(sample_data)
    print("\n-------------------------------\n")
    print("【总结】")
    print("这就解释了为什么它不是原始天气数据：")
    print("1. 真正的天气数据 (HDF5格式) 每一层的形状是 (纬度, 经度)，比如 (128, 256)，记录的是全球各地的实时冷暖。")
    print("2. 这里的 .npz 数据形状通常是单个数值 (或者是一维向量)，它只是记录了所有历史天气数据中某个变量的【全局平均值】或【全局标准差】。")
    print("模型在训练前，需要用这里的平均值去把原始数据进行标准化（Normalization），防止有的数据太大导致梯度爆炸。")
    
    # 画一个简单的柱状图展示前20个变量的均值
    # 如果它是纯标量
    if len(sample_data.shape) == 1 and sample_data.shape[0] == 1:
        plt.figure(figsize=(12, 6))
        
        subset_keys = keys[:20]
        subset_vals = [data[k][0] for k in subset_keys]
        
        plt.bar(subset_keys, subset_vals, color='skyblue')
        plt.xticks(rotation=90)
        plt.title(f"全局均值统计 ({os.path.basename(npz_file_path)}) - 前20个变量")
        plt.tight_layout()
        plt.savefig("npz_visualization.png")
        print("\n✅ 已为您生成了一张这些统计常量的小柱状图: npz_visualization.png")

if __name__ == "__main__":
    npz_path = "normalization_constants/normalize_mean.npz"
    if os.path.exists(npz_path):
        visualize_npz(npz_path)
    else:
        print(f"找不到 {npz_path}")

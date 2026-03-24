import os
import sys

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import h5py
except ImportError as e:
    print("=========================================")
    print("缺少必要的依赖包，请运行以下命令进行安装：")
    print("pip install numpy matplotlib h5py")
    print("或者如果您使用 conda：")
    print("conda install numpy matplotlib h5py")
    print("=========================================")
    print(f"具体报错: {e}")
    sys.exit(1)

def visualize_h5_data(data_path):
    if not os.path.exists(data_path):
        print(f"错误: 找不到文件 {data_path}")
        return

    print(f"正在读取文件: {data_path} ...")
    
    with h5py.File(data_path, 'r') as f:
        # 打印文件层的结构
        print("\n--- HDF5 内部结构 ---")
        for key in f.keys():
            print(f"发现 Group: /{key}")
            if isinstance(f[key], h5py.Group):
                variables = list(f[key].keys())
                print(f"  包含变量数量: {len(variables)}")
                print(f"  前 5 个变量示例: {variables[:5]}")
        
        # 提取其中一个主要变量来进行可视化
        target_group = 'input'
        target_variable = '2m_temperature'
        
        if target_group in f and target_variable in f[target_group]:
            data_matrix = f[target_group][target_variable][:]
            print(f"\n提取变量 '{target_variable}':")
            print(f"  数据形状 (Shape): {data_matrix.shape}")
            print(f"  数据类型 (Dtype): {data_matrix.dtype}")
            print(f"  数据最大值: {np.max(data_matrix):.2f}")
            print(f"  数据最小值: {np.min(data_matrix):.2f}")
            
            # 开始绘图
            plt.figure(figsize=(10, 5))
            # cmap="coolwarm" 比较适合展示气温这种两极化的数据
            im = plt.imshow(data_matrix, cmap='coolwarm', origin='lower')
            plt.colorbar(im, label='Value')
            plt.title(f'Visualization of {target_variable}')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            
            output_img = "real_data_visualization.png"
            plt.tight_layout()
            plt.savefig(output_img, dpi=150)
            print(f"\n✅ 可视化图片已生成，保存在当前目录下的: {output_img}")
        else:
            print(f"\n未在文件中找到 /{target_group}/{target_variable}，请手动修改代码中需要提取的变量名。")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python visualize_real_data.py <你的_h5_文件路径>")
        print("示例: python visualize_real_data.py 2020_0001.h5")
        sys.exit(1)
        
    visualize_h5_data(sys.argv[1])

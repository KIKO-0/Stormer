import os
import numpy as np
import torch
from torchvision.transforms import transforms
from stormer.data.iterative_dataset import ERA5MultiLeadtimeDataset
from stormer.models.iterative_module import GlobalForecastIterativeModule
from stormer.models.hub.stormer import Stormer

"""
这个脚本做什么？
----------------
给定一个“当前时刻”的全球气象场（ERA5），用 Stormer 预测未来时刻（本例是 +6h）的气象场。

整体流程（新手版）:
1) 准备变量列表和设备（CPU/GPU）
2) 构建模型并加载预训练权重
3) 读取归一化统计量（mean/std）和测试样本
4) 配置输入/输出的标准化与反标准化变换
5) 执行自回归推理（roll-out），得到预测结果
6) （可选）在 prediction_dict 上计算指标或可视化

几个关键概念:
- lead time: 目标预测时效（例如 6 小时后、24 小时后）
- base interval: 每次前进一步的时间步长（可选 6/12/24h）
- steps: 为达到目标 lead time 需要滚动多少步（steps = lead_time // interval）
"""

variables = [
    # 与训练保持一致的变量顺序非常关键：
    # - mean/std 的通道顺序
    # - 模型输入输出通道顺序
    # - 可视化和评估时按变量索引取值
    # 三者必须完全一致，否则会出现“变量错位”（例如温度通道被当成风速通道）。
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "geopotential_50",
    "geopotential_100",
    "geopotential_150",
    "geopotential_200",
    "geopotential_250",
    "geopotential_300",
    "geopotential_400",
    "geopotential_500",
    "geopotential_600",
    "geopotential_700",
    "geopotential_850",
    "geopotential_925",
    "geopotential_1000",
    "u_component_of_wind_50",
    "u_component_of_wind_100",
    "u_component_of_wind_150",
    "u_component_of_wind_200",
    "u_component_of_wind_250",
    "u_component_of_wind_300",
    "u_component_of_wind_400",
    "u_component_of_wind_500",
    "u_component_of_wind_600",
    "u_component_of_wind_700",
    "u_component_of_wind_850",
    "u_component_of_wind_925",
    "u_component_of_wind_1000",
    "v_component_of_wind_50",
    "v_component_of_wind_100",
    "v_component_of_wind_150",
    "v_component_of_wind_200",
    "v_component_of_wind_250",
    "v_component_of_wind_300",
    "v_component_of_wind_400",
    "v_component_of_wind_500",
    "v_component_of_wind_600",
    "v_component_of_wind_700",
    "v_component_of_wind_850",
    "v_component_of_wind_925",
    "v_component_of_wind_1000",
    "temperature_50",
    "temperature_100",
    "temperature_150",
    "temperature_200",
    "temperature_250",
    "temperature_300",
    "temperature_400",
    "temperature_500",
    "temperature_600",
    "temperature_700",
    "temperature_850",
    "temperature_925",
    "temperature_1000",
    "specific_humidity_50",
    "specific_humidity_100",
    "specific_humidity_150",
    "specific_humidity_200",
    "specific_humidity_250",
    "specific_humidity_300",
    "specific_humidity_400",
    "specific_humidity_500",
    "specific_humidity_600",
    "specific_humidity_700",
    "specific_humidity_850",
    "specific_humidity_925",
    "specific_humidity_1000",
]

# 自动选择设备：有 CUDA 就用 GPU，否则用 CPU。
# 这里仅用于推理（inference），不是训练。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# 1) 构建模型 + 加载预训练权重
# ============================================================================
# 说明：
# - 这里使用官方提供的 patch_size=4 预训练权重。
# - 如果你换成 patch_size=2，网络结构也必须对应改成 2，否则权重形状不匹配。
pretrained_path = "https://huggingface.co/tungnd/stormer/resolve/main/stormer_1.40625_patch_size_4.ckpt"
net = Stormer(
    in_img_size=[128, 256],  # 1.40625° 全球网格对应约 128(lat) x 256(lon)
    variables=variables,
    patch_size=4,
    hidden_size=1024,
    depth=24,
    num_heads=16,
    mlp_ratio=4,
)
model = GlobalForecastIterativeModule(net, pretrained_path=pretrained_path).to(device)
model.eval()  # 进入推理模式：关闭 dropout，固定 BN/统计行为（若有）

# ============================================================================
# 2) 准备数据：读取标准化统计 + 读取一个测试样本
# ============================================================================
root_dir = "mini_wb2_h5df_regridded"
norm_dir = "normalization_constants"

# 输入标准化统计:
# normalize_mean.npz / normalize_std.npz 中每个变量单独存一份数组。
# 这里按 variables 顺序拼接成通道维数组，形状近似为 [V]，V=变量数(69)。
normalize_mean = dict(np.load(os.path.join(norm_dir, "normalize_mean.npz")))
normalize_mean = np.concatenate([normalize_mean[v] for v in variables], axis=0)
normalize_std = dict(np.load(os.path.join(norm_dir, "normalize_std.npz")))
normalize_std = np.concatenate([normalize_std[v] for v in variables], axis=0)

# 输入变换：x_norm = (x - mean) / std
inp_transform = transforms.Normalize(normalize_mean, normalize_std)

# 这里使用验证/测试数据集类：
# 每个样本返回 (当前输入场, 不同 lead time 的目标场字典, 变量名列表)
dataset = ERA5MultiLeadtimeDataset(
    root_dir=os.path.join(root_dir, "test"),
    variables=variables,
    transform=inp_transform,
    list_lead_times=[6],  # 本示例只取 +6h 目标
    data_freq=6,
)

# 取第 0 个样本:
# inp_data: [V, H, W]（已标准化）
# out_data_dict[lead_time]: [V, H, W]（已标准化）
inp_data, out_data_dict, _ = dataset[0]

# 模型期望 batch 维，因此扩展为 [B, V, H, W]，这里 B=1。
inp_data = inp_data.unsqueeze(0).to(device)
out_data_dict = {k: v.unsqueeze(0).to(device) for k, v in out_data_dict.items()}  # [1, V, H, W]

# ============================================================================
# 3) 配置“差分”尺度变换（针对不同 interval）
# ============================================================================
# Stormer 在迭代时预测的是“增量/差分”（difference），不是绝对值。
# 不同 interval（6h、12h、24h）的差分尺度不同，因此需要不同的 diff_std。
out_transforms = {}
for l in [6, 12, 24]:
    normalize_diff_std = dict(
        np.load(os.path.join(norm_dir, f"normalize_diff_std_{l}.npz"))
    )
    normalize_diff_std = np.concatenate(
        [normalize_diff_std[v] for v in variables], axis=0
    )

    # diff 的标准化形式是:
    # diff_norm = (diff - 0) / diff_std
    # 所以 mean 使用 0，std 使用每个变量对应的 diff_std。
    out_transforms[l] = transforms.Normalize(
        np.zeros_like(normalize_diff_std), normalize_diff_std
    )

# 将输入标准化和差分变换都注册到 module，
# forward_validation 内部会用它们做“标准化空间 <-> 物理空间”的来回切换。
model.set_transforms(inp_transform, out_transforms)

# ============================================================================
# 4) 执行推理（核心）
# ============================================================================
# 思路：
# 对每个目标 lead_time（本例只有 6），尝试所有可整除的 interval。
# 例如 lead_time=24 时：
# - interval=6  -> steps=4
# - interval=12 -> steps=2
# - interval=24 -> steps=1
# 对这些路径的预测结果做平均（ensemble mean）可提升稳健性。
prediction_dict = {}
list_intervals = [6, 12, 24]
for lead_time in out_data_dict.keys():
    all_preds = []
    for interval in list_intervals:
        if lead_time % interval == 0:
            steps = lead_time // interval
            with torch.no_grad():
                # forward_validation 返回标准化空间下的预测:
                # 形状 [B, V, H, W]
                pred = model.forward_validation(inp_data, variables, interval, steps)
            all_preds.append(pred)

    # 将多条 interval 路径进行集成平均。
    mean_pred = torch.stack(all_preds, dim=0).mean(0)  # ensemble mean

    # prediction_dict[lead_time] -> [B, V, H, W]（标准化空间）
    prediction_dict[lead_time] = mean_pred

# ============================================================================
# 5) 后处理（可选）
# ============================================================================
# 你通常会在这里做两件事：
# A. 指标计算：和 out_data_dict[lead_time] 对比（二者都是标准化空间）
# B. 可视化：先反标准化到物理量，再作图
#
# 例如（伪代码）:
# pred_norm = prediction_dict[6]                  # [1, V, H, W]
# gt_norm   = out_data_dict[6]                    # [1, V, H, W]
# rmse_norm = ((pred_norm - gt_norm) ** 2).mean().sqrt()
#
# pred_phys = pred_norm * std + mean              # 逐变量逐像素反标准化
# gt_phys   = gt_norm   * std + mean

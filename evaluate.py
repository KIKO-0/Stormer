import os
import numpy as np
import torch
from torchvision.transforms import transforms
from stormer.data.iterative_dataset import ERA5MultiLeadtimeDataset
from stormer.models.iterative_module import GlobalForecastIterativeModule
from stormer.models.hub.stormer import Stormer
from stormer.utils.metrics import lat_weighted_rmse, lat_weighted_acc

"""
evaluate.py
-----------
在测试集上评估 Stormer 预报精度，输出各变量、各时效的 RMSE 和 ACC。

用法:
    python evaluate.py

修改配置区的 root_dir 指向完整数据集路径后，即可在大规模数据上运行。
目前默认使用 mini 数据集（仅供流程验证）。

输出:
    evaluation_results.npz  — 机器可读结果
    evaluation_results.txt  — 人类可读汇总表
"""

# ============================================================================
# 配置区（按需修改）
# ============================================================================
root_dir  = "mini_wb2_h5df_regridded/test"  # 完整数据集替换此路径
norm_dir  = "normalization_constants"
pretrained_path = "/root/stormer/stormer_1.40625_patch_size_4.ckpt"

# WeatherBench2 标准评估时效（小时）
# mini 数据集只有 8 个时间步，只能评估 6h；完整数据集可用全部时效
val_lead_times = [6]
# 完整数据集时改为：
# val_lead_times = [6, 24, 72, 120, 168]

# 重点评估的变量（WeatherBench2 标准基准变量）
EVAL_VARS = [
    "geopotential_500",         # Z500
    "temperature_850",          # T850
    "2m_temperature",           # T2m
    "10m_u_component_of_wind",  # U10
    "mean_sea_level_pressure",  # MSLP
]

variables = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "geopotential_50", "geopotential_100", "geopotential_150", "geopotential_200",
    "geopotential_250", "geopotential_300", "geopotential_400", "geopotential_500",
    "geopotential_600", "geopotential_700", "geopotential_850", "geopotential_925",
    "geopotential_1000",
    "u_component_of_wind_50", "u_component_of_wind_100", "u_component_of_wind_150",
    "u_component_of_wind_200", "u_component_of_wind_250", "u_component_of_wind_300",
    "u_component_of_wind_400", "u_component_of_wind_500", "u_component_of_wind_600",
    "u_component_of_wind_700", "u_component_of_wind_850", "u_component_of_wind_925",
    "u_component_of_wind_1000",
    "v_component_of_wind_50", "v_component_of_wind_100", "v_component_of_wind_150",
    "v_component_of_wind_200", "v_component_of_wind_250", "v_component_of_wind_300",
    "v_component_of_wind_400", "v_component_of_wind_500", "v_component_of_wind_600",
    "v_component_of_wind_700", "v_component_of_wind_850", "v_component_of_wind_925",
    "v_component_of_wind_1000",
    "temperature_50", "temperature_100", "temperature_150", "temperature_200",
    "temperature_250", "temperature_300", "temperature_400", "temperature_500",
    "temperature_600", "temperature_700", "temperature_850", "temperature_925",
    "temperature_1000",
    "specific_humidity_50", "specific_humidity_100", "specific_humidity_150",
    "specific_humidity_200", "specific_humidity_250", "specific_humidity_300",
    "specific_humidity_400", "specific_humidity_500", "specific_humidity_600",
    "specific_humidity_700", "specific_humidity_850", "specific_humidity_925",
    "specific_humidity_1000",
]

# ============================================================================
# 1) 模型加载
# ============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

net = Stormer(
    in_img_size=[128, 256],
    variables=variables,
    patch_size=4,
    hidden_size=1024,
    depth=24,
    num_heads=16,
    mlp_ratio=4,
)
model = GlobalForecastIterativeModule(net, pretrained_path=pretrained_path).to(device)
model.eval()

# ============================================================================
# 2) 归一化统计量
# ============================================================================
normalize_mean = dict(np.load(os.path.join(norm_dir, "normalize_mean.npz")))
normalize_mean = np.concatenate([normalize_mean[v] for v in variables], axis=0)  # [V]

normalize_std = dict(np.load(os.path.join(norm_dir, "normalize_std.npz")))
normalize_std = np.concatenate([normalize_std[v] for v in variables], axis=0)    # [V]

inp_transform = transforms.Normalize(normalize_mean, normalize_std)

out_transforms = {}
for l in [6, 12, 24]:
    diff_std = dict(np.load(os.path.join(norm_dir, f"normalize_diff_std_{l}.npz")))
    diff_std = np.concatenate([diff_std[v] for v in variables], axis=0)
    out_transforms[l] = transforms.Normalize(np.zeros_like(diff_std), diff_std)

model.set_transforms(inp_transform, out_transforms)

# ============================================================================
# 3) 数据集
# ERA5MultiLeadtimeDataset.__getitem__ 返回:
#   (inp_norm [V,H,W], {lead_time: gt_norm [V,H,W]}, variables)
# inp 和 gt 都已经过 inp_transform 标准化（绝对值场）
# ============================================================================
dataset = ERA5MultiLeadtimeDataset(
    root_dir=root_dir,
    variables=variables,
    transform=inp_transform,
    list_lead_times=val_lead_times,
)

lat_path = os.path.join(os.path.dirname(root_dir), "lat.npy")
lat = np.load(lat_path)  # [H]

# 反标准化 transform: x_phys = x_norm * std + mean
# 用于把模型输出（标准化空间）还原到物理量，再计算 RMSE
inv_transform = transforms.Normalize(
    -normalize_mean / normalize_std,
    1.0 / normalize_std,
)

# 气候态（用于 ACC anomaly 基准），这里用训练均值代替
# 有多年数据时应换为逐日气候态
clim = torch.from_numpy(normalize_mean).view(1, len(variables), 1, 1).float()

# ============================================================================
# 4) 主评估循环
# ============================================================================
rmse_accum = {lt: {v: [] for v in EVAL_VARS} for lt in val_lead_times}
acc_accum  = {lt: {v: [] for v in EVAL_VARS} for lt in val_lead_times}

list_intervals = [6, 12, 24]

print(f"Evaluating {len(dataset)} samples, lead times: {val_lead_times}h")

for sample_idx in range(len(dataset)):
    inp_norm, out_norm_dict, _ = dataset[sample_idx]
    # inp_norm: [V, H, W]，归一化绝对值场
    inp_batch = inp_norm.float().unsqueeze(0).to(device)  # [1, V, H, W]

    for lead_time in val_lead_times:
        if lead_time not in out_norm_dict:
            continue

        gt_norm = out_norm_dict[lead_time].float().unsqueeze(0).to(device)  # [1, V, H, W]

        # ensemble over valid intervals
        all_preds = []
        for interval in list_intervals:
            if lead_time % interval == 0:
                steps = lead_time // interval
                with torch.no_grad():
                    # forward_validation 返回归一化空间的预测 [1, V, H, W]
                    pred_norm = model.forward_validation(inp_batch, variables, interval, steps)
                all_preds.append(pred_norm)
        mean_pred_norm = torch.stack(all_preds, dim=0).mean(0)  # [1, V, H, W]

        # 反标准化到物理空间
        pred_phys = inv_transform(mean_pred_norm.squeeze(0)).unsqueeze(0)  # [1, V, H, W]
        gt_phys   = inv_transform(gt_norm.squeeze(0)).unsqueeze(0)         # [1, V, H, W]

        # RMSE（物理空间，不需要再变换）
        rmse_dict = lat_weighted_rmse(
            pred=pred_phys,
            y=gt_phys,
            transform=lambda x: x,
            vars=variables,
            lat=lat,
            log_postfix=str(lead_time),
        )

        # ACC（物理空间 anomaly）
        # lat_weighted_acc 期望 clim 形状为 [V, H, W]，函数内部会 unsqueeze(0)
        clim_phys = inv_transform(clim.squeeze(0))  # [V, 1, 1]
        clim_phys = clim_phys.expand(len(variables), len(lat), 256).to(device)  # [V, H, W]
        acc_dict = lat_weighted_acc(
            pred=pred_phys,
            y=gt_phys,
            transform=lambda x: x,
            vars=variables,
            lat=lat,
            clim=clim_phys,
            log_postfix=str(lead_time),
        )

        for var in EVAL_VARS:
            rkey = f"w_rmse_{var}_{lead_time}"
            akey = f"acc_{var}_{lead_time}"
            if rkey in rmse_dict:
                rmse_accum[lead_time][var].append(rmse_dict[rkey].item())
            if akey in acc_dict:
                acc_accum[lead_time][var].append(acc_dict[akey].item())

    if (sample_idx + 1) % 10 == 0 or sample_idx == len(dataset) - 1:
        print(f"  {sample_idx + 1}/{len(dataset)} done")

# ============================================================================
# 5) 汇总打印 + 保存
# ============================================================================
print("\n" + "=" * 70)
print(f"{'Variable':<35} {'Lead':>6}h   {'RMSE':>10}   {'ACC':>8}")
print("=" * 70)

save_data = {}
for lt in val_lead_times:
    for var in EVAL_VARS:
        rmse_vals = rmse_accum[lt][var]
        acc_vals  = acc_accum[lt][var]
        if rmse_vals:
            mean_rmse = float(np.mean(rmse_vals))
            mean_acc  = float(np.mean(acc_vals)) if acc_vals else float('nan')
            print(f"{var:<35} {lt:>6}h   {mean_rmse:>10.4f}   {mean_acc:>8.4f}")
            save_data[f"rmse_{var}_{lt}h"] = mean_rmse
            save_data[f"acc_{var}_{lt}h"]  = mean_acc

np.savez("evaluation_results.npz", **save_data)
print("\nSaved: evaluation_results.npz")

lines = [
    f"Stormer (patch_size=4) Evaluation Results",
    f"Data: {root_dir}",
    "=" * 70,
    f"{'Variable':<35} {'Lead':>6}h   {'RMSE':>10}   {'ACC':>8}",
    "=" * 70,
]
for lt in val_lead_times:
    for var in EVAL_VARS:
        kr = f"rmse_{var}_{lt}h"
        ka = f"acc_{var}_{lt}h"
        if kr in save_data:
            lines.append(f"{var:<35} {lt:>6}h   {save_data[kr]:>10.4f}   {save_data[ka]:>8.4f}")

with open("evaluation_results.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("Saved: evaluation_results.txt")

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from torchvision.transforms import transforms

from stormer.data.iterative_dataset import ERA5MultiLeadtimeDataset
from stormer.models.iterative_module import GlobalForecastIterativeModule
from stormer.models.hub.stormer import Stormer

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
pretrained_path = 'https://huggingface.co/tungnd/stormer/resolve/main/stormer_1.40625_patch_size_4.ckpt'
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

# Load data
root_dir = 'mini_wb2_h5df_regridded'
norm_dir = 'normalization_constants'
normalize_mean = dict(np.load(os.path.join(norm_dir, "normalize_mean.npz")))
normalize_mean = np.concatenate([normalize_mean[v] for v in variables], axis=0)
normalize_std = dict(np.load(os.path.join(norm_dir, "normalize_std.npz")))
normalize_std = np.concatenate([normalize_std[v] for v in variables], axis=0)
inp_transform = transforms.Normalize(normalize_mean, normalize_std)

dataset = ERA5MultiLeadtimeDataset(
    root_dir=os.path.join(root_dir, 'test'),
    variables=variables,
    transform=inp_transform,
    list_lead_times=[6],
    data_freq=6,
)

inp_data, out_data_dict, _ = dataset[0]
inp_data_gpu = inp_data.unsqueeze(0).to(device)

out_transforms = {}
for l in [6, 12, 24]:
    normalize_diff_std = dict(np.load(os.path.join(norm_dir, f"normalize_diff_std_{l}.npz")))
    normalize_diff_std = np.concatenate([normalize_diff_std[v] for v in variables], axis=0)
    out_transforms[l] = transforms.Normalize(np.zeros_like(normalize_diff_std), normalize_diff_std)
model.set_transforms(inp_transform, out_transforms)

# Run inference for 6h lead time
with torch.no_grad():
    pred = model.forward_validation(inp_data_gpu, variables, 6, 1)

pred_np = pred.squeeze(0).cpu().numpy()  # (V, H, W)
gt_np = out_data_dict[6].numpy()         # (V, H, W)

# 统一反标准化到物理量空间，避免不同单位/尺度混用导致色条饱和。
inp_np = inp_data.numpy()
inp_denorm = inp_np * normalize_std[:, None, None] + normalize_mean[:, None, None]
pred_denorm = pred_np * normalize_std[:, None, None] + normalize_mean[:, None, None]
gt_denorm = gt_np * normalize_std[:, None, None] + normalize_mean[:, None, None]

var_idx = {v: i for i, v in enumerate(variables)}

PLOT_VARS = [
    ("2m_temperature", "2m Temperature (K)", "RdBu_r"),
    ("geopotential_500", "Geopotential Z500 (m²/s²)", "viridis"),
]

lat = np.linspace(-90 + 1.40625/2, 90 - 1.40625/2, 128)
lon = np.linspace(0, 360, 256, endpoint=False)

fig = plt.figure(figsize=(20, 5 * len(PLOT_VARS)))
proj = ccrs.PlateCarree()

for row, (var_name, title, cmap) in enumerate(PLOT_VARS):
    idx = var_idx[var_name]
    inp_field  = inp_denorm[idx]
    pred_field = pred_denorm[idx]
    gt_field   = gt_denorm[idx]
    err        = pred_field - gt_field
    rmse       = np.sqrt(np.mean(err**2))
    # 使用输入/预测/真值三者联合范围，保证同一变量的三个主图可直接比较。
    vmin = min(inp_field.min(), pred_field.min(), gt_field.min())
    vmax = max(inp_field.max(), pred_field.max(), gt_field.max())
    vabs       = np.abs(err).max()

    panels = [
        (inp_field,  "Input (t=0)",            cmap,       vmin,  vmax),
        (pred_field, "Prediction (+6h)",        cmap,       vmin,  vmax),
        (gt_field,   "Ground Truth (+6h)",      cmap,       vmin,  vmax),
        (err,        f"Error  RMSE={rmse:.3f}", "RdBu_r",  -vabs, vabs),
    ]

    for col, (field, panel_title, pcmap, pmin, pmax) in enumerate(panels):
        ax = fig.add_subplot(
            len(PLOT_VARS), 4, row * 4 + col + 1,
            projection=proj
        )
        im = ax.pcolormesh(lon, lat, field, transform=proj,
                           cmap=pcmap, vmin=pmin, vmax=pmax)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.set_global()
        ax.set_title(f"{title}\n{panel_title}", fontsize=9)
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.04, shrink=0.85)

plt.suptitle("Stormer 6h Forecast  —  2020-01-01 00:00 UTC → 2020-01-01 06:00 UTC",
             fontsize=13, y=1.01)
plt.tight_layout()
out_path = "stormer_inference_visualization.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved to {out_path}")

import os
import numpy as np
import torch
import h5py

from torch.utils.data import Dataset
from glob import glob

def get_data_given_path(path, variables):
    # 每个 h5 文件包含一个时间步的多个变量场，这里按变量顺序拼成 [V, H, W]。
    with h5py.File(path, 'r') as f:
        data = {
            main_key: {
                sub_key: np.array(value) for sub_key, value in group.items() if sub_key in variables + ['time']
        } for main_key, group in f.items() if main_key in ['input']}

    x = [data['input'][v] for v in variables]
    return np.stack(x, axis=0)

def get_out_path(root_dir, year, inp_file_idx, steps):
    # year: current year
    # inp_file_idx: file index of the input in the current year
    # steps: number of steps forward
    out_file_idx = inp_file_idx + steps
    out_path = os.path.join(root_dir, f'{year}_{out_file_idx:04}.h5')
    if not os.path.exists(out_path):
        # 处理跨年边界:
        # 若同一年内索引不存在，先找到当前年可到达的最远步，再跳到下一年继续计数。
        for i in range(steps):
            out_file_idx = inp_file_idx + i
            out_path = os.path.join(root_dir, f'{year}_{out_file_idx:04}.h5')
            if os.path.exists(out_path):
                max_step_forward = i
        remaining_steps = steps - max_step_forward
        next_year = year + 1
        out_path = os.path.join(root_dir, f'{next_year}_{remaining_steps-1:04}.h5')
    return out_path


# training dataset consists of 1 input and multiple desired outputs at multiple intervals (randomly chosen)
class ERA5MultiStepRandomizedDataset(Dataset):
    def __init__(
        self,
        root_dir,
        variables,
        inp_transform,
        out_transform_dict,
        steps,
        list_intervals=[6, 12, 24],
        data_freq=6,
    ):
        super().__init__()
        
        # interval 必须是数据时间分辨率的整数倍（例如 6h 数据频率）。
        for l in list_intervals:
            assert l % data_freq == 0

        self.root_dir = root_dir
        self.variables = variables
        self.inp_transform = inp_transform
        self.out_transform_dict = out_transform_dict
        self.steps = steps
        self.list_intervals = list_intervals
        self.data_freq = data_freq
        
        file_paths = glob(os.path.join(root_dir, '*.h5'))
        file_paths = sorted(file_paths)
        # 末尾样本没有足够未来真值，需裁掉。
        self.inp_file_paths = file_paths[:-(steps * max(list_intervals) // data_freq)]
        self.file_paths = file_paths
        
    def __len__(self):
        return len(self.inp_file_paths)
    
    def __getitem__(self, index):
        inp_path = self.inp_file_paths[index]
        inp_data = get_data_given_path(inp_path, self.variables)

        # 每次随机选择一个基础 interval，提升模型对不同步长的鲁棒性。
        chosen_interval = np.random.choice(self.list_intervals)
        year, inp_file_idx = os.path.basename(inp_path).split('.')[0].split('_')
        year, inp_file_idx = int(year), int(inp_file_idx)
        outs = []
        diffs = []
        last_out = inp_data
        
        # 逐步构造多步监督:
        # 使用“相邻步差分”而不是“相对初始差分”，与模型 roll-out 形式一致。
        for step in range(1, self.steps + 1):
            out_path = get_out_path(self.root_dir, year, inp_file_idx, steps=(step * chosen_interval) // self.data_freq)
            out = get_data_given_path(out_path, self.variables)
            diff = out - last_out
            diff = torch.from_numpy(diff)
            diffs.append(self.out_transform_dict[chosen_interval](diff))
            outs.append(out)
            last_out = out
        
        inp_data = torch.from_numpy(inp_data)
        diffs = torch.stack(diffs, dim=0)
        out_transform_mean = torch.from_numpy(self.out_transform_dict[chosen_interval].mean)
        out_transform_std = torch.from_numpy(self.out_transform_dict[chosen_interval].std)
        list_intervals = np.array([chosen_interval] * self.steps)
        # 将小时单位缩放到十分位（与模型时间嵌入约定保持一致）。
        list_intervals = torch.from_numpy(list_intervals).to(dtype=inp_data.dtype) / 10.0
        
        return (
            self.inp_transform(inp_data), # VxHxW
            diffs, # TxVxHxW
            out_transform_mean, # V
            out_transform_std, # V
            list_intervals,
            self.variables,
        )

# validation and test datasets consist of 1 input and multiple desired outputs at multiple lead times
class ERA5MultiLeadtimeDataset(Dataset):
    def __init__(
        self,
        root_dir,
        variables,
        transform,
        list_lead_times,
        data_freq=6,
    ):
        super().__init__()
        
        # lead time 必须是数据频率的整数倍。
        for l in list_lead_times:
            assert l % data_freq == 0

        self.root_dir = root_dir
        self.variables = variables
        self.transform = transform
        self.list_lead_times = list_lead_times
        self.data_freq = data_freq
        
        file_paths = glob(os.path.join(root_dir, '*.h5'))
        file_paths = sorted(file_paths)
        max_lead_time = max(*list_lead_times) if len(list_lead_times) > 1 else list_lead_times[0]
        max_steps = max_lead_time // data_freq
        # 同样裁掉末尾无真值样本。
        self.inp_file_paths = file_paths[:-max_steps]
        self.file_paths = file_paths
        
    def __len__(self):
        return len(self.inp_file_paths)
    
    def __getitem__(self, index):
        inp_path = self.inp_file_paths[index]
        inp_data = get_data_given_path(inp_path, self.variables)
        year, inp_file_idx = os.path.basename(inp_path).split('.')[0].split('_')
        year, inp_file_idx = int(year), int(inp_file_idx)
        dict_out = {}
        
        # 为每个 lead time 取绝对值目标（不是差分），用于验证阶段的指标计算。
        for lead_time in self.list_lead_times:
            out_path = get_out_path(self.root_dir, year, inp_file_idx, steps=lead_time // self.data_freq)
            dict_out[lead_time] = get_data_given_path(out_path, self.variables)
            
        inp_data = torch.from_numpy(inp_data)
        dict_out = {lead_time: torch.from_numpy(out) for lead_time, out in dict_out.items()}
        
        return (
            self.transform(inp_data), # VxHxW
            {lead_time: self.transform(out) for lead_time, out in dict_out.items()},
            self.variables,
        )

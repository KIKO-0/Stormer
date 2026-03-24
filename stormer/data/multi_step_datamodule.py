import os
from typing import Optional, Sequence, Tuple

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from lightning import LightningDataModule

from stormer.data.iterative_dataset import ERA5MultiStepRandomizedDataset, ERA5MultiLeadtimeDataset


def collate_fn_train(
    batch,
) -> Tuple[torch.tensor, torch.tensor, Sequence[str], Sequence[str]]:
    # 训练批次:
    # - 输入: [B, V, H, W]
    # - 目标差分: [B, T, V, H, W]
    # - 差分标准化参数: [B, V]
    # - 区间序列: [B, T]
    inp = torch.stack([batch[i][0] for i in range(len(batch))]) # B, V, H, W
    out = torch.stack([batch[i][1] for i in range(len(batch))]) # B, T, V, H, W
    out_transform_mean = torch.stack([batch[i][2] for i in range(len(batch))]) # B, V
    out_transform_std = torch.stack([batch[i][3] for i in range(len(batch))]) # B, V
    interval = torch.stack([batch[i][4] for i in range(len(batch))]) # B, T
    variables = batch[0][5]
    return inp, out, out_transform_mean, out_transform_std, interval, variables


def collate_fn_val(batch):
    # 验证/测试批次将不同 lead time 的目标组织成字典:
    # out[lead_time] -> [B, V, H, W]
    inp = torch.stack([batch[i][0] for i in range(len(batch))]) # B, V, H, W

    out_dicts = [batch[i][1] for i in range(len(batch))]
    list_lead_times = out_dicts[0].keys()
    out = {}
    for lead_time in list_lead_times:
        out[lead_time] = torch.stack([out_dicts[i][lead_time] for i in range(len(batch))])
        
    variables = batch[0][2]
    
    return inp, out, variables


class MultiStepDataRandomizedModule(LightningDataModule):
    def __init__(
        self,
        root_dir,
        variables,
        list_train_intervals,
        steps,
        val_lead_times,
        data_freq=6,
        batch_size=64,
        val_batch_size=64,
        num_workers=0,
        pin_memory=False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        # 输入场标准化（按变量通道）。
        normalize_mean = dict(np.load(os.path.join(root_dir, "normalize_mean.npz")))
        normalize_mean = np.concatenate([normalize_mean[v] for v in variables], axis=0)
        normalize_std = dict(np.load(os.path.join(root_dir, "normalize_std.npz")))
        normalize_std = np.concatenate([normalize_std[v] for v in variables], axis=0)
        self.transforms = transforms.Normalize(normalize_mean, normalize_std)
        
        # 不同时间间隔对应不同 diff std（因为步长越大，增量分布尺度通常越大）。
        out_transforms = {}
        for l in list_train_intervals:
            normalize_diff_std = dict(np.load(os.path.join(root_dir, f"normalize_diff_std_{l}.npz")))
            normalize_diff_std = np.concatenate([normalize_diff_std[v] for v in variables], axis=0)
            out_transforms[l] = transforms.Normalize(np.zeros_like(normalize_diff_std), normalize_diff_std)
        self.out_transforms = out_transforms

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def get_lat_lon(self):
        # 经纬度用于纬度加权损失与评估指标。
        lat = np.load(os.path.join(self.hparams.root_dir, "lat.npy"))
        lon = np.load(os.path.join(self.hparams.root_dir, "lon.npy"))
        return lat, lon
    
    def get_transforms(self):
        return self.transforms, self.out_transforms

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            # 训练集: 随机采样 interval + 多步差分监督
            self.data_train = ERA5MultiStepRandomizedDataset(
                root_dir=os.path.join(self.hparams.root_dir, 'train'),
                variables=self.hparams.variables,
                inp_transform=self.transforms,
                out_transform_dict=self.out_transforms,
                steps=self.hparams.steps,
                list_intervals=self.hparams.list_train_intervals,
                data_freq=self.hparams.data_freq,
            )

            # 验证/测试: 固定 lead time 的绝对值目标（用于 roll-out 评估）
            self.data_val = ERA5MultiLeadtimeDataset(
                root_dir=os.path.join(self.hparams.root_dir, 'val'),
                variables=self.hparams.variables,
                transform=self.transforms,
                list_lead_times=self.hparams.val_lead_times,
                data_freq=self.hparams.data_freq
            )

            self.data_test = ERA5MultiLeadtimeDataset(
                root_dir=os.path.join(self.hparams.root_dir, 'test'),
                variables=self.hparams.variables,
                transform=self.transforms,
                list_lead_times=self.hparams.val_lead_times,
                data_freq=self.hparams.data_freq
            )

    def train_dataloader(self):
        # 训练阶段打乱样本，保留最后不完整 batch（drop_last=False）。
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn_train,
        )

    def val_dataloader(self):
        # 验证阶段不打乱，保证可复现评估。
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn_val,
        )

    def test_dataloader(self):
        # 测试阶段与验证保持一致的数据组织方式。
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn_val,
        )

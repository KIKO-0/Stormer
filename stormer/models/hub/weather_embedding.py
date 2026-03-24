from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, trunc_normal_

from stormer.utils.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid


class WeatherEmbedding(nn.Module):
    def __init__(
        self,
        variables,
        img_size,
        patch_size=2,
        embed_dim=1024,
        num_heads=16,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.variables = variables

        # 变量级 tokenization:
        # 每个气象变量（温度/风场/湿度等）使用独立的 PatchEmbed，
        # 这样可保留不同变量的统计差异与语义特性。
        self.token_embeds = nn.ModuleList(
            [PatchEmbed(None, patch_size, 1, embed_dim) for i in range(len(variables))]
        )
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        # 变量身份嵌入（channel embedding）:
        # 标记 token 来自哪个变量，帮助后续变量聚合。
        self.channel_embed, self.channel_map = self.create_var_embedding(embed_dim)

        # 变量聚合:
        # 对每个 patch 位置，使用可学习 query 对所有变量做一次 cross-attention，
        # 将 [V, D] 聚合为单个 [D]，得到统一的 token 序列 [L, D]。
        self.channel_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.channel_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # patch 位置嵌入（二维 sin-cos 初始化，可训练）。
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=True)

        self.initialize_weights()

    def initialize_weights(self):
        # 使用固定 sin-cos 模式初始化空间位置嵌入，给模型提供稳定的相对位置信号。
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        channel_embed = get_1d_sincos_pos_embed_from_grid(self.channel_embed.shape[-1], np.arange(len(self.variables)))
        self.channel_embed.data.copy_(torch.from_numpy(channel_embed).float().unsqueeze(0))

        # PatchEmbed 卷积核初始化（trunc normal）。
        for i in range(len(self.token_embeds)):
            w = self.token_embeds[i].proj.weight.data
            trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        # 其余线性层 / LayerNorm 初始化。
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_var_embedding(self, dim):
        # 创建变量名 -> 索引映射，避免每次 forward 线性扫描变量列表。
        var_embed = nn.Parameter(torch.zeros(1, len(self.variables), dim), requires_grad=True)
        var_map = {}
        idx = 0
        for var in self.variables:
            var_map[var] = idx
            idx += 1
        return var_embed, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        # vars 作为 tuple 可被缓存，减少重复构图与索引开销。
        ids = np.array([self.channel_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb, vars):
        ids = self.get_var_ids(vars, var_emb.device)
        return var_emb[:, ids, :]

    def aggregate_variables(self, x: torch.Tensor):
        """
        x: B, V, L, D
        """
        b, _, l, _ = x.shape
        # [B, V, L, D] -> [B, L, V, D]，然后把 B 和 L 合并，
        # 对每个“样本-位置”独立做变量聚合。
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # BxL, V, D

        var_query = self.channel_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.channel_agg(var_query, x, x)  # BxL, D
        x = x.squeeze()

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D
        return x

    def forward(self, x: torch.Tensor, variables):
        if isinstance(variables, list):
            variables = tuple(variables)

        # 对每个变量通道分别做 patch embedding。
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)

        for i in range(len(var_ids)):
            id = var_ids[i]
            embed_variable = self.token_embeds[id](x[:, i : i + 1]) # B, L, D
            embeds.append(embed_variable)
        x = torch.stack(embeds, dim=1)  # B, V, L, D

        # 同时叠加变量身份嵌入与空间位置嵌入。
        var_embed = self.get_var_emb(self.channel_embed, variables)
        x = x + var_embed.unsqueeze(2)
        x = x + self.pos_embed.unsqueeze(1)

        # 聚合变量维度，输出主干 Transformer 需要的 [B, L, D] token 序列。
        x = self.aggregate_variables(x)  # B, L, D

        return x
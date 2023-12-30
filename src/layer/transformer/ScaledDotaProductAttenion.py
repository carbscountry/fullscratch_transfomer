import numpy as np
import torch
from torch import nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(
        self,
        q: torch.Tensor, # = Q (query)
        k: torch.Tensor, # = K (key)
        v: torch.Tensor, # = V (value)
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        scalar = np.sqart(self.d_k)
        attention_weight = torch.matmul(q, torch.transpose(k,1,2)) / scalar # [Q * K^T] / (D^0.5)
        if mask is not None: # maskに対する処理
            if mask.dim() != attention_weight.dim(): # データの次元数確認のメソッド dim = 次元数
                raise ValueError(
                    "mask.dim != attention_weight.dim, mask.dim={}, attention_weight.dim={}".format(
                        mask.dim(), attention_weight.dim()
                    )
                )
            attention_weight = attention_weight.masked_fill(
                mask, -torch.finfo(torch.float32).max
                ) # maskの部分を-infに置き換える


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from .core import MLP
from .basemodel import BaseModel


class EmbeddingMLPModel(BaseModel):
    """Embedding MLP Model"""
    def __init__(self, feature_columns, label_columns, hidden_units=[128, 128], 
                 l2_reg_linear=1e-5, l2_reg_embedding=1e-5, init_std=0.0001, seed=1024, 
                 task='binary', device='cpu', gpus=None):
        super(EmbeddingMLPModel, self).__init__(feature_columns, label_columns,
                                               l2_reg_linear=l2_reg_linear,
                                               l2_reg_embedding=l2_reg_embedding, 
                                               init_std=init_std,
                                               seed=seed, task=task, 
                                               device=device, gpus=gpus)

        input_dim = self._compute_input_dim()

        # 使用MLP替代原来的DNN
        self.mlp = MLP(input_dim=input_dim, 
                      hidden_units=hidden_units,
                      dropout=0.2,
                      activation='relu',
                      use_bn=False)

        # 输出层,根据不同任务设置不同输出维度
        self.dnn_linear = nn.Linear(hidden_units[-1], len(self.label_columns))

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](x[feat.name].long()) 
                               for feat in self.sparse_feature_columns]
        dense_value_list = [x[feat.name].float() for feat in self.dense_feature_columns]

        dnn_input = torch.cat(sparse_embedding_list + dense_value_list, dim=-1)

        # 使用MLP进行前向传播
        deep_output = self.mlp(dnn_input)
        output = self.dnn_linear(deep_output)

        # 根据不同任务使用不同的输出激活函数
        outputs = {}
        for idx, label_col in enumerate(self.label_columns):
            outputs[label_col.name] = self.out_funcs[label_col.name](output[:, idx:idx+1])
            
        return outputs

    def compile(self,
                optimizer: torch.optim.Optimizer,
                loss: torch.nn.Module = None,
                metrics: list[torch.nn.Module] = None) -> None:
        """Compile the model.

        Args:
            optimizer: The optimizer.
            loss: The loss function.
            metrics: The metrics.
        """
        self.optimizer = optimizer
        # self.loss = loss
        # self.metrics = metrics

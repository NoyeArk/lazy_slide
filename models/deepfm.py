import torch
import torch.nn as nn

from .basemodel import BaseModel
from .core import FM, MLP


class DeepFM(BaseModel):
    """Instantiates the DeepFM Network architecture.

    Args:
        linear_feature_columns: 用于模型线性部分的特征列表。
        dnn_feature_columns: 用于模型深度部分的特征列表。
        use_fm: 是否使用FM层。
        dnn_hidden_units: DNN隐藏层单元数列表,例如[256, 128]。
        l2_reg_linear: 线性部分的L2正则化系数。
        l2_reg_embedding: Embedding向量的L2正则化系数。
        l2_reg_dnn: DNN的L2正则化系数。
        init_std: Embedding向量初始化的标准差。
        seed: 随机种子。
        dnn_dropout: DNN的dropout比率,取值范围[0,1)。
        dnn_activation: DNN使用的激活函数。
        dnn_use_bn: 是否在DNN中使用BatchNormalization。
        task: 任务类型,'binary'用于二分类,'regression'用于回归。
        device: 运行设备,'cpu'或'cuda:0'。
        gpus: GPU设备列表。如果为None,则在device上运行。gpus[0]需要与device相同。

    Returns:
        PyTorch模型实例。
    """

    def __init__(self,
                 feature_columns, label_columns, dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):

        super(DeepFM, self).__init__(feature_columns, label_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)

        self.fm = FM()
        self.dnn = MLP(input_dim=self._compute_input_dim(), 
                       hidden_units=dnn_hidden_units, 
                       dropout=dnn_dropout, 
                       activation=dnn_activation, 
                       use_bn=dnn_use_bn)

        self.to(device)

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](x[feat.name].long()) 
                               for feat in self.sparse_feature_columns]
        dense_value_list = [x[feat.name].float() for feat in self.dense_feature_columns]

        dnn_input = torch.cat(sparse_embedding_list + dense_value_list, dim=-1)

        # FM部分
        if len(sparse_embedding_list) > 0:
            # 将embedding列表重塑为正确的维度 [batch_size, field_size, embedding_size]
            fm_input = torch.stack(sparse_embedding_list, dim=1)  # 使用stack而不是cat
            fm_output = self.fm(fm_input)
        else:
            fm_output = torch.zeros((dnn_input.shape[0], 1), device=dnn_input.device)

        # DNN部分
        deep_output = self.dnn(dnn_input)

        # 将FM和DNN的输出连接起来
        logit = fm_output + deep_output

        # 根据不同任务使用不同的输出激活函数
        outputs = {}
        for idx, label_col in enumerate(self.label_columns):
            outputs[label_col.name] = self.out_funcs[label_col.name](logit)

        return outputs

# import sys
# sys.path.append(sys.path[0] + "/../")

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder
from metric import AUC
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

class BaseModel(nn.Module):
    """Base Model Class, parent class for all models.

    Args:
        linear_feature_columns: List of features used by linear part
        dnn_feature_columns: List of features used by deep network
        l2_reg_linear: float. L2 regularization strength applied to linear part
        l2_reg_embedding: float. L2 regularization strength applied to embedding vectors
        init_std: float. Standard deviation for embedding vector initialization
        seed: int. Random seed
        task: str, "binary" for binary classification or "regression" for regression
        device: str, "cpu" or "cuda:0"
        gpus: List of devices for multiple GPUs. If None, run on device
    """

    def __init__(self, feature_columns, label_columns, l2_reg_linear=1e-5,
                 l2_reg_embedding=1e-5, init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=None):
        super(BaseModel, self).__init__()
        self.dense_feature_columns = [col for col in feature_columns if col.dtype == torch.float32]
        self.sparse_feature_columns = [col for col in feature_columns if col.dtype != torch.float32]
        self.seq_feature_columns = [col for col in feature_columns if col.is_sequence()]

        self.embedding_dict = nn.ModuleDict()
        self.regularization_weight = []
        self.device = device
        self.gpus = gpus
        if gpus and not device.startswith('cuda'):
            raise ValueError('`gpus` list can only be used when device is CUDA')

        # Initialize embedding layers
        for feat in self.sparse_feature_columns + self.seq_feature_columns:
            self.embedding_dict[feat.embedding_name] = nn.Embedding(feat.vocabulary_size, feat.embedding_dim)

        # Set loss function and metrics
        self.to(device)
        self.label_columns = label_columns
        self.loss_funcs = {}
        self.out_funcs = {}
        self.metrics = {}

        for label_col in label_columns:
            # 设置损失函数
            if label_col.loss == 'binary_crossentropy':
                self.out_funcs[label_col.name] = nn.Sigmoid()
                self.loss_funcs[label_col.name] = nn.BCELoss()
            elif label_col.loss == 'masked_mse':
                self.out_funcs[label_col.name] = nn.Identity()
                self.loss_funcs[label_col.name] = nn.MSELoss()
            elif label_col.loss == 'uncertainty_rmse':
                self.out_funcs[label_col.name] = nn.Identity()
                self.loss_funcs[label_col.name] = nn.MSELoss()
            elif label_col.loss == 'uncertainty_bce':
                self.out_funcs[label_col.name] = nn.Sigmoid()
                self.loss_funcs[label_col.name] = nn.BCELoss()
            elif label_col.loss == 'softmax_loss':
                self.out_funcs[label_col.name] = nn.Softmax(dim=1)
                self.loss_funcs[label_col.name] = nn.CrossEntropyLoss()
            else:
                raise ValueError(f"Unsupported loss type: {label_col.loss}")
            
            # 设置评估指标
            self.metrics[label_col.name] = {}
            for metric in label_col.metrics:
                if metric.lower() == 'auc':
                    self.metrics[label_col.name][metric] = AUC()
                elif metric == 'rmse':
                    self.metrics[label_col.name][metric] = nn.RMSE()
                elif metric == 'mae':
                    self.metrics[label_col.name][metric] = nn.MAE()
                elif metric == 'mse':
                    self.metrics[label_col.name][metric] = nn.MSELoss()
                else:
                    raise ValueError(f"不支持的指标类型: {metric}")

        self.metrics_names = ["loss"]

    def compile(self, optimizer, loss=None, metrics=None):
        """Configure the model for training.

        Args:
            optimizer: String or optimizer instance
            loss: String or loss function
            metrics: List of metrics to be evaluated during training
        """
        self.optimizer = optimizer
    
    def _compute_input_dim(self):
        """计算输入维度"""
        input_dim = sum([feat.embedding_dim for feat in self.sparse_feature_columns]) + \
                   sum([1 for feat in self.dense_feature_columns])
        return input_dim

    def fit(self, 
            x: dict[str, np.ndarray], 
            y: dict[str, np.ndarray], 
            epochs: int = 1, 
            initial_epoch: int = 0, 
            split_ratio: float = 0.2, 
            shuffle: bool = True,
            batch_size: int = 32) -> None:
        """Fit the model to the training data.

        Args:
            x: dict[str, torch.Tensor]. The input data.
            y: torch.Tensor. The target data.
            batch_size: int. The batch size.
            epochs: int. The number of epochs.
            verbose: int. The verbosity mode.
            initial_epoch: int. The epoch to start training from.
            validation_split: float. The fraction of the training data to be used as validation data.
            validation_data: tuple[dict[str, torch.Tensor], torch.Tensor]. The validation data.
            shuffle: bool. Whether to shuffle the training data.
        """
        for fea_name, fea_values in x.items():
            if fea_values.dtype.kind in ['O', 'U', 'S']:  # 检查是否为字符串类型
                le = LabelEncoder()
                x[fea_name] = le.fit_transform(fea_values)
                print(f"对特征 {fea_name} 进行了LabelEncoder编码")

        x_tensor = {k: torch.tensor(v.astype(np.float32)) for k, v in x.items()}
        y_tensor = {k: torch.tensor(v.astype(np.float32)) for k, v in y.items()}
        
        # 将字典转换为张量列表
        x_tensor_list = [x_tensor[feat.name] for feat in self.sparse_feature_columns + self.dense_feature_columns]
        y_tensor_list = [y_tensor[label.name] for label in self.label_columns]
        
        # 创建数据集
        dataset = TensorDataset(*x_tensor_list, *y_tensor_list)

        # 划分训练集和验证集
        dataset_size = len(dataset)
        val_size = int(dataset_size * split_ratio)
        train_size = dataset_size - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        if val_size > 0:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            validation_data = val_loader

        for epoch in range(initial_epoch, epochs):
            self.train()
            total_loss = 0
            n_batches = 0
            
            # 创建进度条对象
            pbar = tqdm(train_loader, desc=f'epoch {epoch+1}/{epochs}', ncols=100)
            for batch in pbar:
                # 分离特征和标签
                x_batch = {feat.name: batch[i].to(self.device) 
                          for i, feat in enumerate(self.sparse_feature_columns + self.dense_feature_columns)}
                y_batch = {label.name: batch[i + len(self.sparse_feature_columns + self.dense_feature_columns)].to(self.device)
                          for i, label in enumerate(self.label_columns)}
                
                y_pred = self(x_batch)

                loss = 0
                for label_col in self.label_columns:
                    target = y_batch[label_col.name].float().view(-1, 1)
                    loss += self.loss_funcs[label_col.name](y_pred[label_col.name], 
                                                          target) * label_col.loss_weight

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1
                
                # 更新进度条后缀显示当前loss
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            if validation_data is not None:
                self.eval()
                val_loss = 0
                val_batches = 0
                # 初始化每个指标的累积值
                metric_values = {label_col.name: {metric: 0 for metric in label_col.metrics} 
                               for label_col in self.label_columns}
                
                with torch.no_grad():
                    # 使用tqdm创建验证进度条
                    val_pbar = tqdm(validation_data, desc='验证中', ncols=100)
                    for batch in val_pbar:
                        x_batch = {feat.name: batch[i].to(self.device) 
                                 for i, feat in enumerate(self.sparse_feature_columns + self.dense_feature_columns)}
                        y_batch = {label.name: batch[i + len(self.sparse_feature_columns + self.dense_feature_columns)].to(self.device)
                                 for i, label in enumerate(self.label_columns)}
                        
                        val_pred = self(x_batch)
                        batch_val_loss = 0
                        
                        # 计算每个标签的损失和指标
                        for label_col in self.label_columns:
                            target = y_batch[label_col.name].float().view(-1, 1)
                            pred = val_pred[label_col.name]
                            
                            # 计算损失
                            curr_loss = self.loss_funcs[label_col.name](pred, target).item() * label_col.loss_weight
                            batch_val_loss += curr_loss
                            
                            # 计算所有支持的指标
                            for metric_name, metric_func in self.metrics[label_col.name].items():
                                metric_values[label_col.name][metric_name] += metric_func(pred, target).item()
                        
                        val_loss += batch_val_loss
                        val_batches += 1
                        
                        # 更新验证进度条
                        val_pbar.set_postfix({'val_loss': f'{batch_val_loss:.4f}'})

                # 计算并打印平均损失和指标
                avg_val_loss = val_loss / val_batches
                print(f'\n验证集评估结果:')
                print(f'平均损失: {avg_val_loss:.4f}')

                # 打印每个标签的指标
                for label_col in self.label_columns:
                    print(f'\n{label_col.name}的评估指标:')
                    for metric in label_col.metrics:
                        avg_metric = metric_values[label_col.name][metric] / val_batches
                        print(f'- {metric}: {avg_metric:.4f}')

    def evaluate(self, x, y, batch_size=256):
        """Evaluate the model.

        Args:
            x: Feature input
            y: Target labels
            batch_size: int. Number of samples per batch
            
        Returns:
            Dictionary containing evaluation results
        """
        pass

    def predict(self, x, batch_size=256):
        """Generate predictions.

        Args:
            x: Feature input
            batch_size: int. Number of samples per batch
            
        Returns:
            Model predictions
        """
        pass

    def _get_loss(self, outputs, labels):
        """Calculate loss value.

        Args:
            outputs: Model outputs
            labels: True labels
            
        Returns:
            Loss value
        """
        return self.loss_func(outputs, labels)

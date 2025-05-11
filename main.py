import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from trainer import Trainer
from datasets.movielens import MovieDataset
from models.embedding_mlp import EmbeddingMLPModel


def main():
    # 数据加载
    train_dataset = MovieDataset("./data/train_samples.csv")
    test_dataset = MovieDataset("./data/test_samples.csv")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 如果有验证集，加载验证集
    val_dataset_path = "./data/val_samples.csv"
    val_loader = None
    try:
        val_dataset = MovieDataset(val_dataset_path)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        print("验证集已加载")
    except FileNotFoundError:
        print("未找到验证集，将跳过验证过程")

    # 模型、损失函数和优化器
    input_size = train_dataset[0][0].shape[0]
    model = EmbeddingMLPModel(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练器
    trainer = Trainer(model, train_loader, val_loader, test_loader, criterion, optimizer, device="cuda" if torch.cuda.is_available() else "cpu")

    # 训练和测试
    trainer.train(epochs=5)
    trainer.test()


if __name__ == "__main__":
    main()

import torch


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, criterion, optimizer, device="cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader  # 验证集可以为 None
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                x, y = [item.to(self.device) for item in batch]
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}, Training Loss: {total_loss / len(self.train_loader)}")
            if self.val_loader:
                self.validate()

    def validate(self):
        if not self.val_loader:
            print("没有验证集，跳过验证过程")
            return

        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.val_loader:
                x, y = [item.to(self.device) for item in batch]
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                total_loss += loss.item()

                predictions = (outputs > 0.5).float()
                correct += (predictions == y).sum().item()
                total += y.size(0)

        print(f"Validation Loss: {total_loss / len(self.val_loader)}, Validation Accuracy: {correct / total:.2%}")

    def test(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.test_loader:
                x, y = [item.to(self.device) for item in batch]
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                total_loss += loss.item()

                predictions = (outputs > 0.5).float()
                correct += (predictions == y).sum().item()
                total += y.size(0)

        print(f"Test Loss: {total_loss / len(self.test_loader)}, Test Accuracy: {correct / total:.2%}")

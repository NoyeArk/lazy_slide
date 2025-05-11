import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


class AUC(nn.Module):
    """AUC metric"""
    def __init__(self):
        super(AUC, self).__init__()
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate AUC score
        
        Args:
            y_pred: Predicted values, shape [batch_size, 1]
            y_true: True values, shape [batch_size, 1]
            
        Returns:
            torch.Tensor: AUC score
        """
        # Convert tensors to numpy arrays
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        
        # Ensure input is a one-dimensional array
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)
        
        # Use sklearn's roc_auc_score to calculate AUC
        try:
            auc_score = roc_auc_score(y_true, y_pred)
            return torch.tensor(auc_score, device=y_pred.device)
        except ValueError:
            # Handle special cases (e.g., all labels are the same)
            return torch.tensor(0.5, device=y_pred.device)

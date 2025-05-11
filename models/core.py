import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-Layer Perceptron Module
    
    Args:
        - input_dim: Input dimension
        - hidden_units: List of hidden layer units, e.g. [256, 128, 64]
        - dropout: Dropout rate, default 0.0
        - activation: Activation function, default 'relu'
        - use_bn: Whether to use BatchNorm, default False
    """
    def __init__(self, input_dim, hidden_units, dropout=0.0, activation='relu', use_bn=False):
        super(MLP, self).__init__()
        
        self.dropout = dropout
        self.activation = activation
        self.use_bn = use_bn

        # Build MLP layers
        layers = []
        hidden_units = [input_dim] + hidden_units
        
        for i in range(len(hidden_units) - 1):
            layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
            
            if self.use_bn:
                layers.append(nn.BatchNorm1d(hidden_units[i + 1]))
                
            if self.activation.lower() == 'relu':
                layers.append(nn.ReLU())
            elif self.activation.lower() == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif self.activation.lower() == 'tanh':
                layers.append(nn.Tanh())
                
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
                
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, inputs):
        return self.mlp(inputs)


class FM(nn.Module):
    """Factorization Machine Module
    
    Module for computing second-order cross features, using formula:
    y = sum((sum(vi*vj)xi*xj))
    where vi,vj are embedding vectors
    """
    def __init__(self):
        super(FM, self).__init__()

    def forward(self, inputs):
        """
        Args:
            inputs: 3D tensor with shape [batch_size, field_size, embedding_size]
        """
        # Calculate square of sum
        square_of_sum = torch.sum(inputs, dim=1) ** 2
        # Calculate sum of square
        sum_of_square = torch.sum(inputs ** 2, dim=1)
        # According to FM formula, second order term is (square_of_sum - sum_of_square) * 0.5
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * cross_term

        return torch.sum(cross_term, dim=1, keepdim=True)

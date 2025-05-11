import sys
sys.path.append(sys.path[0] + "/../")

import torch
import pandas as pd
from torch.utils.data import DataLoader

from models.embedding_mlp import EmbeddingMLPModel
from models.deepfm import DeepFM
from input import create_feature_columns, create_label_columns


if __name__ == "__main__":
    EMBEDDING_DIM = 32
    feature_columns_config = [
        {"name": "fc_i_iv_itemid", "source_feature": "movieId", "vocabulary_size": 1001},
        {"name": "fc_i_iv_userid", "source_feature": "userId", "vocabulary_size": 30001},
        {"name": "fc_i_iv_moviegenre1", "source_feature": "movieGenre1", "vocabulary_size": 1001},
        {"name": "fc_i_iv_moviegenre2", "source_feature": "movieGenre2", "vocabulary_size": 1001},
        {"name": "fc_i_iv_moviegenre3", "source_feature": "movieGenre3", "vocabulary_size": 1001},
    ]
    label_columns_config = [
        {
            "name": "is_click",
            "source_label": "label", 
            "operation": "greater_equal", 
            "max_threshold": 1, 
            "loss": "binary_crossentropy", 
            "loss_weight": 1, 
            "metrics": ["auc"]
        },
    ]

    feature_columns = create_feature_columns(feature_columns_config, EMBEDDING_DIM)
    label_columns = create_label_columns(label_columns_config)

    data = pd.read_csv("./data/train_samples.csv")

    x = {fc.name: data[fc.source_feature].values for fc in feature_columns}
    y = {lc.name: data[lc.source_label].values for lc in label_columns}

    model = DeepFM(feature_columns, label_columns, dnn_hidden_units=[64, 32, 16, 8, 1])
    model.compile(optimizer=torch.optim.Adam(model.parameters()))
    model.fit(x, y, epochs=10)

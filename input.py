import torch
from collections import namedtuple, OrderedDict
from torch.nn.init import normal_

DEFAULT_GROUP_NAME = "default_group"

class FeatureColumn(namedtuple('FeatureColumn', [
    'name', 'source_feature', 'embedding_dim', 'maxlen', 'embedding_name', 'hash_type', 
    'trainable', 'group_name', 'combiner', 'is_id_feature', 'vocabulary_size'])):

    VALID_PREFIX = 'fc'
    # user, item, context, cross
    VALID_SOURCES = {'u', 'i', 'c', 'x'}
    # id, float, sequence id, sequence float
    VALID_OUTPUT_TYPES = {'iv', 'fv', 'isv', 'fsv'} 
    DTYPE_MAPPING = {
        'iv': torch.int64,
        'isv': torch.int64,
        'fv': torch.float32,
        'fsv': torch.float32
    }

    def __new__(cls, name, source_feature, embedding_dim, maxlen, embedding_name=None, 
                hash_type=None, trainable=True, group_name=DEFAULT_GROUP_NAME, 
                combiner="mean", is_id_feature=False, vocabulary_size=None):
        names = name.split('_')
        if len(names) < 4:
            raise ValueError(f"Feature column name '{name}' is invalid: must have at least 4 parts")
        if names[0] != cls.VALID_PREFIX:
            raise ValueError(f"Feature column name '{name}' is invalid: must start with '{cls.VALID_PREFIX}'")
        if names[1] not in cls.VALID_SOURCES:
            raise ValueError(f"Feature column source type '{names[1]}' is invalid: must be one of {cls.VALID_SOURCES}")

        output_type = names[2]
        if output_type not in cls.VALID_OUTPUT_TYPES:
            raise ValueError(f"Feature column output type '{output_type}' is invalid: must be one of {cls.VALID_OUTPUT_TYPES}")
        if output_type in {'isv', 'fsv'} and maxlen <= 1:
            raise ValueError(f"Feature column with output type '{output_type}' must have maxlen > 1")
        
        dtype = cls.DTYPE_MAPPING[output_type]
        if dtype == torch.int64:
            is_id_feature = True
        if dtype == torch.int64 and hash_type is None:
            hash_type = 'lookup'
        if dtype == torch.int64 and embedding_name is None:
            embedding_name = name

        instance = super(FeatureColumn, cls).__new__(cls, name, source_feature, embedding_dim, maxlen, 
                                                      embedding_name, hash_type, trainable, 
                                                      group_name, combiner, is_id_feature, vocabulary_size)
        instance.dtype = dtype
        return instance

    def is_sequence(self):
        return self.maxlen > 1 and self.dtype == torch.int64
    
    def is_polling(self):
        return 'mean' in self.combiner or 'sum' in self.combiner or 'max' in self.combiner
    
    def is_user_fc(self):
        return self.name.split('_')[1] == 'u'
    
    def is_item_fc(self):
        return self.name.split('_')[1] == 'i'
    
    def is_context_fc(self):
        return self.name.split('_')[1] == 'c'
    
    def is_cross_fc(self):
        return self.name.split('_')[1] == 'x'
    
    def is_din(self):
        return self.is_sequence() and self.slot == 2 and self.is_user_fc()
    
    def is_single_spase(self):
        return (not self.is_sequence()) and self.dtype == torch.int64


class LabelColumn(namedtuple('LabelColumn', ['name', 'operation', 'max_threshold', 'min_threshold', 'loss', 'metrics', 'loss_weight', 'source_label', 'scale'])):
    __slots__ = ()
    VALID_OPERATIONS = {'greater_equal','less_equal', 'clipping', 'one_hot', 'logarithm'}
    DEFAULT_LOSS_METRICS = {
        'binary_crossentropy': ['auc', 'label_avg'],
        'masked_mse': ['masked_rmse'],
        'uncertainty_rmse': ['masked_rmse'],
        'ad_ocpc_masked_bce': ['auc', 'label_avg'],
        'uncertainty_bce': ['auc'],
        'softmax_loss': ['accuracy'],
        'reduce_mean': [],
        'in_batch_softmax_loss_by_sample_weight': ['mrr', 'diag_hits@10', 'diag_hits@50', 'diag_hits@100'],
        'in_batch_softmax_loss': ['mrr', 'diag_hits@10', 'diag_hits@50', 'diag_hits@100'],
        'in_batch_softmax_loss_with_negative_sample': ['mrr', 'diag_hits@10', 'diag_hits@50', 'diag_hits@100'],
        'in_batch_softmax_loss_standard': ['mrr', 'diag_hits@10', 'diag_hits@50', 'diag_hits@100'],
    }

    def __new__(cls, name, source_label, operation='greater_equal', max_threshold=1, min_threshold=0, loss='binary_crossentropy', loss_weight=1.0, metrics=None, scale=1.0):
        cls._validate_source_label(source_label)
        cls._validate_operation(operation)
        metrics = metrics or cls.DEFAULT_LOSS_METRICS.get(loss, [])
        return super(LabelColumn, cls).__new__(cls, name, operation, max_threshold, min_threshold, loss, metrics, loss_weight, source_label, scale)
    
    @classmethod
    def _validate_source_label(cls, source_label):
        if not source_label:
            raise ValueError("source_label cannot be None and must refer to a valid label")

    @classmethod
    def _validate_operation(cls, operation):
        if operation not in cls.VALID_OPERATIONS:
            raise ValueError(f"Unsupported operation '{operation}'. Must be one of {cls.VALID_OPERATIONS}")

def build_input_features(feature_columns):
    input_features = OrderedDict()
    for column in feature_columns:
        if column.maxlen > 1:
            input_features[column.name] = (column.maxlen,)
        else:
            input_features[column.name] = ()
    return input_features

def create_feature_columns(feature_columns_config: list[dict], embedding_dim: int) -> list[FeatureColumn]:
    feature_columns = []
    for config in feature_columns_config:
        try:
            feature_column = FeatureColumn(
                name=config['name'],
                source_feature=config['source_feature'],
                embedding_dim=config.get('embedding_dim', embedding_dim),
                maxlen=config.get('maxlen', 1),
                embedding_name=config.get('embedding_name', None),
                group_name=config.get('group_name',DEFAULT_GROUP_NAME),
                is_id_feature=False,
                vocabulary_size=config.get('vocabulary_size', None)
            )
            feature_columns.append(feature_column)
            print(f"Created FeatureColumn: {feature_column}")
        except ValueError as e:
            print(e)
    print(">>> create_feature_columns", feature_columns)

    return feature_columns

def create_label_columns(label_columns_config: list[dict]) -> list[LabelColumn]:
    label_columns = []
    for config in label_columns_config:
        try:
            label_column = LabelColumn(
                name=config['name'],
                source_label=config['source_label'],
                operation=config['operation'],
                loss=config['loss'],
                metrics=config.get('metrics'),  
                loss_weight=config.get('loss_weight', 1.0),
                scale=config.get('scale', 1.0),
            )
            label_columns.append(label_column)
            print(f"Created LabelColumn: {label_column}")

            if config['loss'] == 'wce_loss' or config['loss'] == 'softmax_loss':
                if config['loss'] == 'softmax_loss':
                    op = 'clipping'
                else:
                    op = config['operation']
                print(">>> softmax_loss")
                label_column = LabelColumn(
                    name=config['name'] + '_serving',
                    source_label=config['source_label'],
                    operation=op,
                    max_threshold=config['max_threshold'],
                    min_threshold=config.get('min_threshold', 0.0),
                    loss='zero_loss',
                    metrics=[],  
                    loss_weight=0,
                    scale=1,
                )
                label_columns.append(label_column)
                print(f"Created LabelColumn: {label_column}")
        except ValueError as e:
            print(e)
    print(">>> create_label_columns", label_columns)
    return label_columns

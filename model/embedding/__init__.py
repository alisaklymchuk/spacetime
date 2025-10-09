from .base import Embedding
from .linear import LinearEmbedding
from .repeat import RepeatEmbedding
from .linear_mod import ModifiedLinearEmbedding


def init_embedding(config):
    methods = ['linear', 'linear_mod', 'identity', 'repeat']
    if config['method'] == 'linear':
        return LinearEmbedding(**config['kwargs'])
    elif config['method'] == 'linear_mod':
        return ModifiedLinearEmbedding(**config['kwargs'])
    elif config['method'] == 'repeat':
        return RepeatEmbedding(**config['kwargs'])
    elif config['method'] == 'identity' or config['method'] is None:
        return Embedding(**config['kwargs'])
    else:
        raise NotImplementedError(f'Embedding method {config["method"]} not implemented. Please select among {methods}')
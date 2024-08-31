# optimizers/optimizer_params.py

from torch.optim import Adam, AdamW, SGD

optimizer_configs = {
    'adam': {
        'class': Adam,
        'params': {
            'lr': 2e-5,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.01
        }
    },
    'adamw': {
        'class': AdamW,
        'params': {
            'lr': 2e-5,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.01
        }
    },
    'sgd': {
        'class': SGD,
        'params': {
            'lr': 0.01,
            'momentum': 0.9,
            'weight_decay': 0.0001
        }
    }
}

def get_optimizer_config(optimizer_name):
    return optimizer_configs.get(optimizer_name.lower())
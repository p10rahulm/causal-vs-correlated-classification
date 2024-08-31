# optimizers/optimizer_params.py

from torch.optim import Adam, AdamW, NAdam

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
            'lr': 3e-4,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.01
        }
    },
    'nadam': {
        'class': NAdam,
        'params': {
            'lr': 2e-3,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.01
        }
    }
}

def get_optimizer_config(optimizer_name):
    config = optimizer_configs.get(optimizer_name.lower())
    return config.copy() if config else None
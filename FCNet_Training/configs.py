model_configuration= {
    'resnet50': {
        'crop_num': 512,
        'combine_num': 4096,
        'feature_num': 2048,
        'feature_size': 512,
        'policy_hidden_dim': 256,
        'feature_map_channels': 512,
    }
}

train_configuration= {
    'resnet50': {
        'momentum': 0.9,
        'epoch_num': 100,
        'batch_size': 64,
        'image_size': 96,
        'num_workers': 16,
        "classes_num": 200,
        'RL_epoch_num': 50,
        'weight_decay': 5e-4,
        'learning_rate': 0.01,
    }
}
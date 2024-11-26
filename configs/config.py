config_CELEBA = {
    "dataset_name": "CELEBA",
    "image_shape": (3, 64, 64),
    "batch_size": 64,
    "epochs": 500,
    "learning_rate": 3e-4,
    "project_name": "CELEBA_wandb",
    "log_sample_interval": 5,
    "wandb_name": "Test",
    "data_dir": "cropped_celeba_bin",  # Path to your .bin files

}

config_CIFAR10 = {
    "dataset_name": "CIFAR10",
    "image_shape": (3, 32, 32),  
    "batch_size": 128,
    "epochs": 500,
    "learning_rate": 3e-4,
    "project_name": "CIFAR10_wandb",
    "log_sample_interval": 5,
    "wandb_name": "Test",
}

config_MNIST = {
    "dataset_name": "MNIST",
    "image_shape": (1, 32, 32),
    "batch_size": 128,
    "epochs": 500,
    "learning_rate": 3e-4,
    "project_name": "MNIST_wandb",
    "log_sample_interval": 5,
    "wandb_name": "Test",
}
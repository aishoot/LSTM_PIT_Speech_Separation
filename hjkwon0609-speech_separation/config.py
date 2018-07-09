class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    num_final_features = 513

    batch_size = 16 # 16
    output_size = num_final_features * 2
    num_hidden = 128

    num_layers = 3

    num_epochs = 50
    l2_lambda = 0.0000001
    lr = 5e-4
class Config():
    learning_rate = 0.003
    adam_betas = (0.9, 0.999)
    epoch_count = 10
    batch_size = 32
    experiment_name = 'mnist_mil_1'
    
    train_bag_count = 2048
    val_bag_count = 256
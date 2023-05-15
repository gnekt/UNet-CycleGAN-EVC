import os
import shutil
import os.path as osp
from munch import Munch
import yaml
import random
from trainer import train
import argparse
import wandb


def parser():
    parser = argparse.ArgumentParser(description='Example parser')
    parser.add_argument('--num_worker', type=int, help='Number of workers', default='1')

    args = parser.parse_args()

    num_worker = args.num_worker
    print(f'Number of workers: {num_worker}')
    return args

if __name__ == "__main__":
    args = parser()
    
    # Configs reader
    config = Munch(yaml.safe_load(open("config.yml")))
    log_dir = config['log_dir']
    

    # wandb.init(
    #         # set the wandb project where this run will be logged
    #         project="Sad-UnetCycleGan-2d1d2dUnet",
    #         # track hyperparameters and run metadata
    #         config={
    #         "learning_rate": config.training_parameter["learning_rate"],
    #         "weight_decay": config.training_parameter["weight_decay"],
    #         "architecture": config.log_dir.split("/")[-1],
    #         "dataset": "SadConversion",
    #         "epochs": config.training_parameter["epochs"],
    #         "batch_size": config.training_parameter["batch_size"],
    #         "dropout": config.training_parameter["dropout"],
    #         "sampling":"Half,Half,Half",
    #         "kernel_size": "(3,3),(3,3),(3,3)",
    #         "seq_len": 192,
    #         "features_map": "64,128,256",
    #         "lambda_identity": config.training_parameter["lambda_identity"],
    #         "lambda_cycle": config.training_parameter["lambda_cycle"]
    #         }
    #     )
    # initialize wandb
    # start a new wandb run to track this script
    
    # train the model
    train("config.yml", args.num_worker)
    
    wandb.finish()

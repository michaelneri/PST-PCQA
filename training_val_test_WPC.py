from data import WPCModule
from model import PST_PCQAModule
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy import stats
import torch
import numpy as np
import pandas as pd
import os
import shutil

def train(datamodule, model, configs):
    wandb_logger = WandbLogger(project="NRPCQA", config = configs,  name=configs["name"], tags = ["WPC"])
    checkpoint_callback = ModelCheckpoint(
        dirpath = "best_models_WPC/", 
        filename = "PST_PCQAModule-WPC-GVP-{epoch:02d}-{val/loss_mse:.4f}",  # <--- note epoch suffix here
        save_last = True, 
        every_n_epochs = 1,
        save_top_k = 10,
        monitor= "val/loss_mse",
        auto_insert_metric_name=False
    ) 

    shutil.rmtree("best_models_WPC/", ignore_errors=True)
    os.mkdir("best_models_WPC/")

    # definition of the trainer 
    trainer = Trainer(accelerator="gpu", devices = 1 , max_epochs = configs["epochs"], logger=wandb_logger, callbacks = [checkpoint_callback])
    wandb_logger.watch(model, log_graph=False)
    trainer.fit(model, datamodule)
    return model, trainer

def scan_all_dir(path):
        list_all_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                list_all_files.append(str(root + "\\" + file))
        return list_all_files

def test(trained_model:PST_PCQAModule, datamodule, trainer:Trainer, configs:dict):
    # for metrics 
    list_checkpoints = scan_all_dir("best_models_WPC/")
    best_PLCC = -1000
    best_checkpoint = ""
    for i, checkpoint in enumerate(list_checkpoints):
        print("{}) {}".format(i, checkpoint))
        trained_model = PST_PCQAModule.load_from_checkpoint(checkpoint,  points_texture = configs["points_texture"],
                                                         points_structure = configs["points_structure"], dropout = configs["dropout"], patches=configs['number_patches'], lr = configs["lr"])
        trained_model.predicted_mos = []
        trained_model.true_mos = []
        trainer.test(trained_model, datamodule)
            


        pred_mos = torch.cat(trained_model.predicted_mos).flatten().cpu().numpy()
        true_mos = torch.cat(trained_model.true_mos).flatten().cpu().numpy()

        PLCC = stats.mstats.pearsonr(true_mos, pred_mos)[0]   # PLCC
        SRCC = stats.mstats.spearmanr(true_mos, pred_mos)[0]  # SRCC
        KRCC = stats.mstats.kendalltau(true_mos, pred_mos)[0] # KRCC
        RMSE = np.sqrt(np.mean((pred_mos*datamodule.max_mos - true_mos*datamodule.max_mos)**2)) # RMSE

        if PLCC > best_PLCC:
            best_PLCC = PLCC
            best_checkpoint = checkpoint
            print("Best PLCC : {}".format(best_PLCC))
            
    trained_model = PST_PCQAModule.load_from_checkpoint(best_checkpoint,  points_texture = configs["points_texture"],
                                                         points_structure = configs["points_structure"], dropout = configs["dropout"], patches=configs['number_patches'], lr = configs["lr"])
    trainer.test(trained_model, datamodule)
    pred_mos = torch.cat(trained_model.predicted_mos).flatten().cpu().numpy()
    true_mos = torch.cat(trained_model.true_mos).flatten().cpu().numpy()

    PLCC = stats.mstats.pearsonr(true_mos, pred_mos)[0]   # PLCC
    SRCC = stats.mstats.spearmanr(true_mos, pred_mos)[0]  # SRCC
    KRCC = stats.mstats.kendalltau(true_mos, pred_mos)[0] # KRCC
    RMSE = np.sqrt(np.mean((pred_mos*datamodule.max_mos - true_mos*datamodule.max_mos)**2))

    np.savez("pred_mos", pred_mos)
    np.savez("true_mos", true_mos)

    wandb.log({"test/PLCC": PLCC})
    wandb.log({"test/SRCC": SRCC})
    wandb.log({"test/KRCC": KRCC})
    wandb.log({"test/RMSE": RMSE})
    wandb.finish()


if __name__ == "__main__":
    # for each dataset and for each performance evaluation process
    info_file = pd.read_excel("data/WPC/mos.xls")
    path_ply = "data/WPC/distorted"
    configs = {
        "name": "PST_PCQAModule",
        "number_patches": 16,
        "points_per_patch" : 14900,
        "points_texture" : 8192,
        "points_structure" : 1024,
        "batch_size" : 2,
        "lr" : 0.001,
        "dropout": 0.,
        "epochs": 400
    }

    datamodule = WPCModule(pd_db = info_file, path_ply = path_ply, number_patches = configs["number_patches"], points_structure = configs['points_structure'], batch_size = configs["batch_size"])
    model = PST_PCQAModule(points_texture = configs["points_texture"] , points_structure = configs["points_structure"], dropout = configs["dropout"], patches=configs['number_patches'], lr = configs["lr"])
    trained_model, trainer = train(datamodule, model, configs)
    test(trained_model, datamodule, trainer, configs)








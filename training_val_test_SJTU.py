from data import SJTU_PCQAModule
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

def train(configs, k):
    info_file = pd.read_excel("data/SJTU-PCQA/MOS.xlsx", header = None)
    path_ply = "data/SJTU-PCQA/distorted"
    datamodule = SJTU_PCQAModule(pd_db = info_file, path_ply = path_ply, number_patches = configs["number_patches"], points_structure=configs['points_structure'], batch_size = configs["batch_size"], fold = k)
    wandb_logger = WandbLogger(project = "NRPCQA", config = configs,  name=configs["name"] + "Fold {}".format(k), tags = ["SJTU"])
    model = PST_PCQAModule(points_texture = configs["points_texture"] , points_structure = configs["points_structure"], dropout = configs["dropout"], patches=configs['number_patches'], lr = configs["lr"])
    
    checkpoint_callback = ModelCheckpoint(
        dirpath = "best_models_SJTU/Fold{}".format(k), 
        filename = "SJTU-GVP-"+str(k)+"Fold-{epoch:02d}-{val/loss_mse:.4f}",  # <--- note epoch suffix here
        save_last = True, 
        every_n_epochs = 1,
        save_top_k = 5,
        monitor= "val/loss_mse",
        auto_insert_metric_name = False
    ) 

    shutil.rmtree("best_models_SJTU/Fold{}".format(k))
    os.mkdir("best_models_SJTU/Fold{}".format(k))
    

    # definition of the trainer 
    trainer = Trainer(accelerator = "gpu", devices = 1 , max_epochs = configs["epochs"], logger = wandb_logger, callbacks = [checkpoint_callback])
    wandb_logger.watch(model, log_graph = False)
    trainer.fit(model, datamodule)
    return model, datamodule, trainer

def scan_all_dir(path):
        list_all_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                list_all_files.append(str(root + "\\" + file))
        return list_all_files

def test(trained_model:PST_PCQAModule, datamodule, trainer:Trainer, configs:dict, k):
    # for metrics 
    list_checkpoints = scan_all_dir("best_models_SJTU/Fold{}".format(k))
    best_PLCC = -1000
    best_checkpoint = ""
    for i, checkpoint in enumerate(list_checkpoints):
        print("{}) {}".format(i, checkpoint))
        trained_model = PST_PCQAModule.load_from_checkpoint(checkpoint, points_texture = configs["points_texture"],
                                                         points_structure = configs["points_structure"], dropout = configs["dropout"], patches=configs['number_patches'], lr = configs["lr"])
        trained_model.predicted_mos = []
        trained_model.true_mos = []
        trainer.test(trained_model, datamodule)
            
        pred_mos = torch.cat(trained_model.predicted_mos).flatten().cpu().numpy()
        true_mos = torch.cat(trained_model.true_mos).flatten().cpu().numpy()

        PLCC = stats.mstats.pearsonr(true_mos*datamodule.max_mos, pred_mos*datamodule.max_mos)[0]   # PLCC
        SRCC = stats.mstats.spearmanr(true_mos*datamodule.max_mos, pred_mos*datamodule.max_mos)[0]  # SRCC
        KRCC = stats.mstats.kendalltau(true_mos*datamodule.max_mos, pred_mos*datamodule.max_mos)[0] # KRCC
        RMSE = np.sqrt(np.mean((pred_mos*datamodule.max_mos - true_mos*datamodule.max_mos)**2))

        if PLCC > best_PLCC:
            best_PLCC = PLCC
            best_checkpoint = checkpoint
            print("Best : {} %".format(best_PLCC))
            
    trained_model = PST_PCQAModule.load_from_checkpoint(best_checkpoint,  points_texture = configs["points_texture"],
                                                         points_structure = configs["points_structure"], dropout = configs["dropout"], patches=configs['number_patches'], lr = configs["lr"])
    trainer.test(trained_model, datamodule)
    pred_mos = torch.cat(trained_model.predicted_mos).flatten().cpu().numpy()
    true_mos = torch.cat(trained_model.true_mos).flatten().cpu().numpy()

    PLCC = stats.mstats.pearsonr(true_mos*datamodule.max_mos, pred_mos*datamodule.max_mos)[0]   # PLCC
    SRCC = stats.mstats.spearmanr(true_mos*datamodule.max_mos, pred_mos*datamodule.max_mos)[0]  # SRCC
    KRCC = stats.mstats.kendalltau(true_mos*datamodule.max_mos, pred_mos*datamodule.max_mos)[0] # KRCC
    RMSE = np.sqrt(np.mean((pred_mos*datamodule.max_mos - true_mos*datamodule.max_mos)**2))

    np.savez("pred_mos", pred_mos)
    np.savez("true_mos", true_mos)

    wandb.log({"test/PLCC": PLCC})
    wandb.log({"test/SRCC": SRCC})
    wandb.log({"test/KRCC": KRCC})
    wandb.log({"test/RMSE": RMSE})
    wandb.finish()
    return PLCC, SRCC, KRCC, RMSE


if __name__ == "__main__":
    # for each dataset and for each performance evaluation process
    configs = {
        "name": "PST_PCQAModule",
        "number_patches": 16,
        "points_per_patch" : 14900,
        "points_texture" : 8192,
        "points_structure" : 1024,
        "batch_size" : 4,
        "lr" : 0.001,
        "dropout": 0.,
        "epochs": 400
    }

    
    # K = 9 leave-out cross validation
    all_PLCC, all_SRCC, all_KRCC, all_RMSE = [], [], [], []
    for k in range(3, 9):
        trained_model, datamodule, trainer = train(configs, k)
        PLCC, SRCC, KRCC, RMSE = test(trained_model, datamodule, trainer, configs, k)
        all_PLCC.append(PLCC)
        all_SRCC.append(SRCC)
        all_KRCC.append(KRCC)
        all_RMSE.append(RMSE)

    all_PLCC = np.array(all_PLCC)
    all_SRCC = np.array(all_SRCC)
    all_KRCC = np.array(all_KRCC)
    all_RMSE = np.array(all_RMSE)

    np.savez("data/SJTU-PCQA/all_PLCC", all_PLCC)
    np.savez("data/SJTU-PCQA/all_SRCC", all_SRCC)
    np.savez("data/SJTU-PCQA/all_KRCC", all_KRCC)
    np.savez("data/SJTU-PCQA/all_RMSE", all_RMSE)

    wandb_logger = WandbLogger(project="NRPCQA", config = configs,  name=configs["name"] + "All results", tags = ["SJTU"])

    wandb.log({"test/PLCC": np.mean(all_PLCC)})
    wandb.log({"test/SRCC": np.mean(all_SRCC)})
    wandb.log({"test/KRCC": np.mean(all_KRCC)})
    wandb.log({"test/RMSE": np.mean(all_RMSE)})

    wandb.finish()

     








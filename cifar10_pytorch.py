import cifar10_datamodule # get data
import cifar10_train # get model
from pathlib import Path
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning import Trainer



########################
###  Add Argument Parser
########################

# Argument parser for user defined paths
parser = ArgumentParser()

parser.add_argument(
    "--tensorboard_root",
    type=str,
    default="output/tensorboard",
    help="Tensorboard Root path (default: output/tensorboard)",
)

parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="output/train/models",
    help="Path to save model checkpoints (default: output/train/models)",
)

parser.add_argument(
    "--dataset_path",
    type=str,
    default="./",
    help="Cifar10 Dataset path (default: ./)",
)

parser.add_argument(
    "--ptl_args",
    type=str,
    default="max_epochs=1, gpus=0, accelerator=None, profiler='simple', gradient_clip_val=1",
    help="Arguments specific to PTL trainer"
)

parser.add_argument("--trial_id", default=0, type=int, help="Trial id")


########################
###  Trainer Arguments defined via arguments
########################

args = vars(parser.parse_args())
ptl_args = args["ptl_args"]
trial_id = args["trial_id"]

TENSORBOARD_ROOT = args["tensorboard_root"]
CHECKPOINT_DIR = args["checkpoint_dir"]
DATASET_PATH = args["dataset_path"]

# ptl_args = "max_epochs=3, gpus=0, accelerator=None, profiler=pytorch"
ptl_dict = eval("dict({})".format(ptl_args))


########################
###  Callbacks
########################
 
lr_logger = LearningRateMonitor()
tboard = TensorBoardLogger(TENSORBOARD_ROOT)
early_stopping = EarlyStopping(
    monitor="val_loss", mode="min", patience=5, verbose=True
)
checkpoint_callback = ModelCheckpoint(
    dirpath=CHECKPOINT_DIR,
    filename="cifar10_{epoch:02d}",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min",
)

########################
###  Trainer Arguments defined explicitly
########################
trainer_args = {
    "logger": tboard,
    "enable_checkpointing": True,
    "callbacks": [lr_logger, early_stopping, checkpoint_callback],
}

# # Trainer Arguments defined via arguments
if "accelerator" in ptl_dict and ptl_dict["accelerator"] == "None":
    ptl_dict["accelerator"] = None

if not ptl_dict["max_epochs"]:
    trainer_args["max_epochs"] = 3
else:
    trainer_args["max_epochs"] = ptl_dict["max_epochs"]

if "profiler" in ptl_dict and ptl_dict["profiler"] != "":
    trainer_args["profiler"] = ptl_dict["profiler"]


# Setting the datamodule specific arguments
data_module_args = {"train_glob": DATASET_PATH}

# Creating parent directories
Path(TENSORBOARD_ROOT).mkdir(parents=True, exist_ok=True)
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

trainer_args.update(ptl_dict)

data_module = cifar10_datamodule.CIFAR10DataModule(**data_module_args)
model = cifar10_train.CIFAR10Classifier()

trainer = Trainer(
    **trainer_args
)

lr_finder = trainer.tuner.lr_find(model, datamodule=data_module, early_stop_threshold=None, min_lr=0.0001)
lr = lr_finder.suggestion()
print(f"Found lr: {lr}")
model.lr = lr


trainer.fit(model, datamodule=data_module)  
trainer.test(model, datamodule=data_module)

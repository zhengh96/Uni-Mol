import pickle
from unimol_tools import MolTrain
import random

data = pickle.load(open("../data/23pep_data.pkl", "rb"))

savedir = "./23pep_model"

clf = MolTrain(
    task = "multilabel_regression",
    data_type = "molecule",
    epochs = 100, 
    batch_size = 512,
    metrics = ["mse"],
    learning_rate=1e-4,
    early_stopping=10,
    save_path=savedir,
    kfold=4,
    gpu_id=0,
    loss_key="mse",
    label_weight=[0.1, 0.9],
)

clf.fit(data=data)
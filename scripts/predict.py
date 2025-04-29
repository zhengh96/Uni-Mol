import pickle
from unimol_tools import MolPredict
import random

data = pickle.load(open("../data/23pep_data.pkl", "rb"))
to_pred_data = pickle.load(open("../data/4pep_data.pkl", "rb"))

savedir = "./23pep_model"

clf = MolPredict(data = savedir)
pred = clf.predict(data=to_pred_data).tolist()

result_score = {}
temp_result = {}

for pep, score in zip(to_pred_data["seq"], pred):
    temp_result[pep] = temp_result.get(pep, []) + [score]

for pep, score in temp_result.items():
    result_score[pep] = [sum([s[0] for s in score])/len(score), sum([s[1] for s in score])/len(score)]

with open("predict_result.csv", "w") as f:
    f.write("Peptide,Pred_AP,Pred_SHB\n")
    for pep, score in result_score.items():
        f.write(f"{pep},{score[0]:.4f},{score[1]:.4f}\n")
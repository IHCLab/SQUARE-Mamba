import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
from functions.util import *
from networks.SQUARE_Mamba import SQUARE_Mamba

current_directory = os.getcwd()
file_path = f"{current_directory}/main"
os.chdir(file_path)

def test(test_gen, model):
  model.eval()
  prediction = []
  with torch.no_grad():
    for iteration, (Data) in enumerate(test_gen):
      Data = Data.to("cuda")
      spei = model.forward(Data)
      prediction.append(spei.cpu().squeeze(1))
  return np.array(prediction)


if __name__ == '__main__':
  
  model = SQUARE_Mamba(in_channel=105)
  model = model.to("cuda")
  data_Pooncarie, gt_Pooncarie = load_data(1260, 1476)
  testing_data, testing_gt = Create_dataset(data_Pooncarie, gt_Pooncarie, num_sample=201)
  testloader = DataLoader(testing_data, batch_size=201, shuffle=False)
    
  folder_path = "./checkpoint/SQUARE_Mamba.pkl"
  model.load_state_dict(torch.load(folder_path))
  prediction_temp = test(testloader, model)
  gt_test = testing_gt[8:201, 4]
  prediction = prediction_temp[:, 8:201]

  gt_csv, prediction_csv = pd.DataFrame(gt_test.reshape(-1, 1)), pd.DataFrame(prediction.reshape(-1, 1))
  gt_csv.to_csv("./Result/SQUARE_Mamba/gt_Pooncarie.csv", header=None, index=False)
  prediction_csv.to_csv("./Result/SQUARE_Mamba/prediction_Pooncarie.csv", header=None, index=False)
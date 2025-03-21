import torch.optim as optim
import argparse  
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from importlib import import_module
import numpy as np
from tqdm import tqdm
from functions.util import *
import torch.nn as nn

current_directory = os.getcwd()
file_path = f"{current_directory}/main"
os.chdir(file_path)

parser = argparse.ArgumentParser(description="PyTorch Train")
parser.add_argument("--start_epoch", type=int,default=1, help="Start epoch from 1")
parser.add_argument('--model', default='SQUARE_Mamba',type=str, help='Import which network')
parser.add_argument('--lr', default=1e-3, help='initial learning rate')
training_settings = [{'nEpochs': 251, 'start_epoch': 1}]

def validate(val_gen, model, epoch, best_loss_R2):
  model.eval()
  val_loss_list, val_loss_list_r2 = [], []

  with torch.no_grad():
    for iteration, (Data, gt) in tqdm(enumerate(val_gen)):
         
      gt = gt.to('cuda')
      Data = Data.to("cuda")
      spei = model.forward(Data)
      loss = loss_function(gt, spei)
      
      val_loss_list.append(loss.item())
      val_loss = np.array(val_loss_list).mean()

      loss_r2 = r_square(gt, spei)
      val_loss_list_r2.append(loss_r2.item())
      val_loss_r2 = np.array(val_loss_list_r2).mean()

      print("===>Epoch{} Part: Validation loss is :{:4f}" .format(epoch, val_loss))

    if best_loss_R2 < val_loss_r2:
      best_loss_R2 = val_loss_r2
      torch.save(model.state_dict(), "./checkpoint/SQUARE_Mamba.pkl")

  return val_loss, best_loss_R2


def train(train_gen, model, optimizer, epoch):
  model.train()
  train_loss_list=[]

  for iteration, (Data, gt) in tqdm(enumerate(train_gen)):
      
    optimizer.zero_grad()
    gt = gt.to('cuda')
    Data = Data.to("cuda")
    spei = model.forward(Data)
    
    loss = loss_function(gt, spei)
    loss.backward()
    
    optimizer.step()
    train_loss_list.append(loss.item())
    train_loss = np.array(train_loss_list).mean()
        
    if iteration % 1 == 0:
      print("===> Epoch[{}]({}/{}): Loss{:.4f};".format(epoch,iteration, len(trainloader), loss))
    print("===>Epoch{} Part: Avg loss is :{:4f}".format(epoch, train_loss))

  return train_loss

if __name__ == '__main__':
  
  training_data, gt_training = load_data(0, 960)
  validation_data, gt_validation = load_data(960, 1260)
  training_dataset, train_gt = Create_dataset(training_data, gt_training, num_sample=945)
  validation_dataset, val_gt = Create_dataset(validation_data, gt_validation, num_sample=285)

  loss_function = nn.MSELoss()
  Loss_list, Val_Loss_list, best_loss_r2 = [], [], -999
  opt = parser.parse_args()
  Net = import_module('networks.' + opt.model)
  seed = 42
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  model = Net.make_model().to("cuda")
  num_epoch = training_settings[0]['nEpochs']

  optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.0001)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch*3, eta_min=1e-7)
  print("===> Loading model {} and criterion".format(opt.model))
  
  opt.nEpochs = training_settings[0]['nEpochs']
  opt.start_epoch = training_settings[0]['start_epoch']
  trainloader = DataLoader(TensorDataset(training_dataset, train_gt[:, 4].reshape(-1, 1)), batch_size=315, shuffle=False)
  valloader = DataLoader(TensorDataset(validation_dataset, val_gt[:, 4].reshape(-1, 1)), batch_size=285, shuffle=False)

  for epoch in range(opt.start_epoch, opt.nEpochs):
    trainloss = train(trainloader, model, optimizer, epoch)
    Loss_list.append(trainloss.item())
    if epoch % 1 == 0:
      valloss, best_loss_r2 = validate(valloader, model, epoch, best_loss_r2)
      Val_Loss_list.append(valloss.item())
    scheduler.step()
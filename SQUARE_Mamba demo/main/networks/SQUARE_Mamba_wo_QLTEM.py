import os
import torch
import torch.nn as nn
from mamba_ssm import Mamba

current_directory = os.getcwd()
file_path = f"{current_directory}"
os.chdir(file_path)
##
def make_model():
  return SQUARE_Mamba(in_channel=105)


def map_generation(spei_tensor, channels):

  batch_size = spei_tensor.shape[0]
  pixel_vector = spei_tensor.reshape(batch_size, 9, -1)
  map = torch.zeros((batch_size, 3, 3, channels)).to("cuda")

  map[:, 0, 0:3, :] = pixel_vector[:, 0:3, :]
  map[:, 1, 0:3, :] = pixel_vector[:, 3:6, :]
  map[:, 2, 0:3, :] = pixel_vector[:, 6:9, :]

  return map.permute(0, 3, 1, 2) 


def nearest_padding(input_tensor):
    
  batch, channel = input_tensor.shape[0], input_tensor.shape[1]
  padding_tensor = input_tensor
  mask = padding_tensor[0, 0] == 0 
  non_zero_idx = torch.nonzero(~mask, as_tuple=True)
  zero_idx = torch.nonzero(mask, as_tuple=True)
  distances = (non_zero_idx[0][None, :] - zero_idx[0][:, None]) ** 2 + (non_zero_idx[1][None, :] - zero_idx[1][:, None]) ** 2
  
  nearest_idx = distances.argmin(dim=1)
  padding_values = padding_tensor[:batch, :channel, non_zero_idx[0][nearest_idx], non_zero_idx[1][nearest_idx]]
  padding_tensor[:batch, :channel, zero_idx[0], zero_idx[1]] = padding_values

  return padding_tensor
         

class ConvLayer(nn.Module):
  def __init__(self, d_model):
    super(ConvLayer, self).__init__()
    self.Conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1, padding_mode="circular")
    self.norm = nn.BatchNorm1d(d_model)
    self.activation = nn.ELU()

  def forward(self, x):
      
    x = self.norm(x.permute(0, 2, 1))
    x = self.Conv(x)
    x = self.activation(x)
    x = x.transpose(1,2)
    return x

class auxiliary_decoder(nn.Module):
  def __init__(self, d_model, time_step, dropout=0.2):
    super(auxiliary_decoder, self).__init__()

    self.d_ff = 128
    self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=self.d_ff, kernel_size=1)
    self.conv2 = nn.Conv1d(in_channels=self.d_ff, out_channels=d_model, kernel_size=1)
    self.norm = nn.BatchNorm1d(time_step)
    self.dropout = nn.Dropout(dropout)
    self.linear = nn.Linear(d_model, 1)
    self.linear2 = nn.Linear(time_step, 1)
    self.activation = nn.GELU()

  def forward(self, x):
      
    y = x
    y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
    y = self.dropout(self.conv2(y).transpose(-1,1))
    output = self.linear(self.norm(x+y))
    output = (self.linear2(output.transpose(-1, 1))).transpose(-1, 1)

    return output.squeeze(1)

class spatial_encoding_block(nn.Module): 
  def __init__(self, in_channel):
    super(spatial_encoding_block, self).__init__()
    
    self.in_channel = in_channel
    self.depthwise_conv = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=2, groups=self.in_channel)
    self.leakyReLU = nn.LeakyReLU(negative_slope=0.2)
    self.max_pooling = nn.MaxPool2d(2)

  def forward(self, augmented_tensor_temp):
    
    augmented_tensor = nearest_padding(augmented_tensor_temp)
    feature_local = self.max_pooling(self.leakyReLU(self.depthwise_conv(augmented_tensor)))
    feature_local = feature_local.squeeze()+augmented_tensor[:, :, 1, 1]
          
    return feature_local.reshape(-1, 15, 7)


class LTEM(nn.Module):
  def __init__(self):
    super(LTEM, self).__init__()

    self.mamba_1 = Mamba(d_model=7, d_state=32, d_conv=3, expand=20,)
    self.mamba_2 = Mamba(d_model=7, d_state=32, d_conv=3, expand=20,)
    self.mamba_3 = Mamba(d_model=7, d_state=32, d_conv=3, expand=20,)
    self.mamba_4 = Mamba(d_model=7, d_state=32, d_conv=3, expand=20,)
    self.mamba_5 = Mamba(d_model=7, d_state=32, d_conv=3, expand=20,)

    self.conv1 = ConvLayer(d_model=7)
    self.conv2 = ConvLayer(d_model=7)
    self.conv3 = ConvLayer(d_model=7)
    self.conv4 = ConvLayer(d_model=7)
    self.conv5 = ConvLayer(d_model=7)

  def forward(self, x):
    
    local_temporal_group = torch.chunk(x, 5, dim=1)

    group_1 = local_temporal_group[0]  
    output_1 = self.conv1(self.mamba_1(group_1)) 
    
    group_2 = local_temporal_group[1]
    output_2 = self.conv2(self.mamba_2(group_2)) 
    
    group_3 = local_temporal_group[2]
    output_3 = self.conv3(self.mamba_3(group_3)) 
    
    group_4 = local_temporal_group[3]
    output_4 = self.conv4(self.mamba_4(group_4) ) 

    group_5 = local_temporal_group[4]
    output_5 = self.conv5(self.mamba_5(group_5)) 
    
    return output_1, output_2, output_3, output_4, output_5 

class feature_fusion_block(nn.Module):
  def __init__(self):
    super(feature_fusion_block, self).__init__()
    
    self.PDM = auxiliary_decoder(d_model=7, time_step=15)

  def forward(self, group_1, group_2, group_3, group_4, group_5):
    
    decoder_input = torch.concat([group_1, group_2, group_3, group_4, group_5], dim=1)
    predicted_SPEI = self.PDM(decoder_input)
    
    return predicted_SPEI

class SQUARE_Mamba(nn.Module):
  def __init__(self, in_channel):
    super(SQUARE_Mamba, self).__init__()

    self.in_channel = in_channel
    self.SEB = spatial_encoding_block(self.in_channel)
    self.LTEM = LTEM()
    self.FFB = feature_fusion_block()
    self.tanh = nn.Tanh()

  def forward(self, x):
    
    augmented_tensor = map_generation(x, self.in_channel) 
    spatial_feature = self.SEB(augmented_tensor) 
    
    ltem_1, ltem_2, ltem_3, ltem_4, ltem_5 = self.LTEM(spatial_feature)

    spei = 3*self.tanh(self.FFB(ltem_1, ltem_2, ltem_3, ltem_4, ltem_5))
    return spei
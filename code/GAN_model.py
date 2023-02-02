"""
SenseGAN module. Written by Faiga Alawad.
"""
import os
import torch
from torch import nn

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
torch.cuda.empty_cache()

def get_noise( seq_len, features_dim, device):
    """
    Function for getting noise Tensor to the Generator.
    """
    # pylint: disable=E1101
    return  torch.randn(1, seq_len, features_dim, device=device)

class Generator(nn.Module):
    """
    Class for the Generator.
    """
    def __init__(self, e_output, device, seq_dim):
        super().__init__()

        self.num_layers = 3
        self.hidden_dim = 128
        self.device = device
        self.e_output = e_output
        self.seq_dim = seq_dim

        self.lstm = nn.LSTM( input_size = e_output, hidden_size = self.hidden_dim,
                  num_layers = self.num_layers, batch_first = True)
        
        self.dense = nn.Sequential(
          nn.Linear(self.hidden_dim , e_output),
          )

    def forward(self, noise):
        """
        Function for completing a forward pass of the Generator.
        """
        h0 = torch.rand(self.num_layers, noise.shape[0], self.hidden_dim, device=self.device)
        c0 = torch.rand(self.num_layers, noise.shape[0], self.hidden_dim, device=self.device)
        gen_data_, _ = self.lstm(noise, (h0, c0))
        out = self.dense(gen_data_)
        return out

class Discriminator(nn.Module):
    """
    The Discriminator class.
    """
    def __init__(self, e_output, device, seq_dim):
        super().__init__()

        self.num_layers = 1
        self.hidden_dim = 128
        self.device = device

        self.lstm = nn.LSTM(
                  input_size = e_output,
                  hidden_size = self.hidden_dim,
                  num_layers = self.num_layers,
                  bidirectional = True,
                  batch_first = True )
                

        self.dense = nn.Sequential(
           nn.Linear(self.hidden_dim * 2 , 1),
           nn.Flatten(),
           nn.Linear(seq_dim, 1),
           nn.Flatten(start_dim=0))

    def forward(self, gen_data):
        """
        Function for completing a forward pass of the Discriminator.
        """
        h0 = torch.rand(self.num_layers * 2, gen_data.shape[0], self.hidden_dim, device=self.device)
        c0 = torch.rand(self.num_layers * 2, gen_data.shape[0], self.hidden_dim, device=self.device)
        out1, _ = self.lstm(gen_data, (h0, c0))
        out = self.dense(out1)
        return out

"""
This module will plot the results, given specific folder.
"""
import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchinfo import summary

from sklearn.metrics import mean_squared_error


from data_loading import real_data_loading
from GAN_model import get_noise

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

columns_room_ocup = ['Temperature', 'CO2', 'Humidity']
columns_dic = {'stock': ['1', '2', '3', '4', '5', '6'],
                'all_intel': ['Temperature', 'Humidity', 'Light', 'Voltage'],
                'energy':  list(np.arange(1, 29).astype(str)),
                'sine': ['1']}

def plot_losses(folder):
    """
    Plot the generator and discriminator losses on the same graph.
    """
    csv_files = glob.glob(folder + "*.csv")
    loss_fun_csv_files =sorted(csv_files, key=os.path.getmtime)
    all_loss_fun = pd.DataFrame()
    for file_name in loss_fun_csv_files:
        column_name = file_name[11] + '_loss'
        all_loss_fun[column_name]  = pd.read_csv(file_name)
    all_loss_fun = all_loss_fun.reindex(sorted(all_loss_fun.columns), axis=1)
    
    d_loss = all_loss_fun[all_loss_fun.columns[0]]
    g_loss = all_loss_fun[all_loss_fun.columns[1]]

    _, _ = plt.subplots(figsize=(7,3.5), dpi=700)
    plt.plot(d_loss, label = 'dloss')
    plt.plot(g_loss, label = 'gloss')
    #plt.title(all_loss_fun.columns[i][2:].upper())
    #plt.ylim(-1.5,2.5)
    plt.legend()
    plt.savefig(folder + "losses.png", bbox_inches='tight', dpi=700)

def plot_helper(fake_df, ori_data, model_name, folder):
    """
    This helper will get the no of columns and rows for subplot,
    and plot the fake versus the original data.
    """
    num_subplots = len(fake_df.columns)
    rows = int(np.sqrt(num_subplots))
    cols = int(num_subplots/ rows)
    if rows * cols != num_subplots:
        rows += 1

    fig, axes = plt.subplots(rows, cols, figsize=(15,7), dpi=700)

    axes_list = []
    for ax_ in axes:
        axes_list.append(ax_)

    for i, col in enumerate(fake_df.columns):
        axes_list[i].plot(fake_df[col], label ='Fake')
        axes_list[i].plot(ori_data[col][:1000], label ='Real')
        axes_list[i].title.set_text(str(col))

        _rmse = mean_squared_error(ori_data[col][:1000], fake_df[col], squared = False)
        _rmse_text = 'RMSE = ' + str(round(_rmse,2))
        axes_list[i].text(0.5, 0.95, _rmse_text, horizontalalignment='center',
                            verticalalignment='center', transform=axes_list[i].transAxes,
                            bbox=dict(facecolor='white', alpha=0.2))
        axes_list[i].legend()

    plt.suptitle(model_name)
    fig.savefig(folder + model_name + '_.png', bbox_inches='tight', dpi=700)

def plot_generated_data(folder):
    """
    Plot generated vs real data on the same graph.
    """
    pkl_files = glob.glob(folder + "*.pkl")
    for file_name in pkl_files:
        model_name = file_name[-9:-4]
    
        _ , transformer, ori_data = real_data_loading(100)

        ori_data = pd.DataFrame(ori_data, columns=columns_room_ocup)
        generator_ = torch.load(file_name, map_location ='cuda')
        summary(generator_)
        generated_data = generator_(get_noise(1000, len(columns_room_ocup) , 'cuda'))
        generated_data = generated_data.cpu().detach().numpy()
        generated_data = generated_data.reshape(generated_data.shape[1], generated_data.shape[2])

        fake_df = transformer.inverse_transform(generated_data)
        fake_df = pd.DataFrame(fake_df, columns=columns_room_ocup)

        plot_helper(fake_df, ori_data, model_name, folder)


def main(args):
    """
    main is the default function for the compiler, which runs
    the other helper functions in this file.
    """
    plot_losses(args.folder)
    plot_generated_data(args.folder)

if __name__ == '__main__':
    parser_ = argparse.ArgumentParser()
    parser_.add_argument('--folder', type=str)
    args_ = parser_.parse_args()
    main(args_)

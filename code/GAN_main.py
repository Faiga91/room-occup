"""
The main module for senseGAN, that train the GAN network.
"""
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

# pylint: disable=E0401
import torch

# pylint: disable=E0401
from data_loading import real_data_loading, sine_data_generation

from GAN_model import get_noise
from GAN_model import Generator, Discriminator


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SCRIPT_DIR = os.path.dirname(os.path.realpath('__file__')) + 'Code/'
sys.path.append(os.path.dirname(SCRIPT_DIR))


def main(args):
    """
    Main function for SenseGAN
    Args:
      * data_name: sine, stock, or energy, all_intel.
      * seq_len: Sequence length.
      * n_epochs: numbers of training epochs.
      * learning_rate
      * device: 'cuda' for GPU usage or 'cpu' for CPU usage.
      * model_name: the name of the trained Generator model to save for later use.
    Return:
     * Save the generator model to a Pikle file.
    """
    ## Data loading
    if args.data_name == 'sine_wave':
        _, ori_data = sine_data_generation(samples_no = 1, seq_len = 1e6)
    else:
        ori_data, _ , _ = real_data_loading(args.data_name, args.seq_len,
                                                    scaler_name=args.scaler_name)

    ori_data = torch.FloatTensor(np.asarray(ori_data))

    print('\n The model name is ', args.model_name)

    _, _, feature_dim = ori_data.shape
    trainldr = torch.utils.data.DataLoader(dataset=ori_data[:-4],
                                              batch_size=args.batch_size,
                                              shuffle=False)


    # pylint: disable=E1102
    gen = Generator(feature_dim, args.device, args.seq_len).to(args.device)
    disc = Discriminator(feature_dim, args.device, args.seq_len).to(args.device)
    #gen.train()
    #disc.train()

    gen_opt = torch.optim.Adam(gen.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    disc_opt = torch.optim.Adam(disc.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    g_losses = []
    d_losses = []

    gloss_fun = dispatcher[args.loss_fun + '_gloss']
    dloss_fun = dispatcher[args.loss_fun + '_dloss']

    # Start the training loop
    for epoch in range(args.n_epochs):
        for real in tqdm(trainldr, leave=False):
            real = real.to(args.device).float()
            disc_opt.zero_grad()
            fake_noise = get_noise(args.seq_len, feature_dim, device=args.device)
            fake = gen(fake_noise).detach()
            crit_fake_pred = disc(fake)
            crit_real_pred = disc(real)

            if args.loss_fun in ['wgan_gp', 'dragan']:
                with torch.backends.cudnn.flags(enabled=False):
                    epsilon = torch.rand(real.shape[0], real.shape[1], real.shape[2],
                                        device=args.device, requires_grad=True)
                    gradient = get_gradient(disc, real, fake, epsilon)
                    if args.loss_fun == 'wgan_gp':
                        grad_penalty= gradient_penalty(gradient)
                    if args.loss_fun == 'dragan':
                        gp_s = deep_regret_gradient(args.batch_size, real, disc, args.device)
                        grad_penalty= torch.mean(gp_s)
                    disc_loss = dloss_fun(crit_fake_pred, crit_real_pred, grad_penalty, 1)
            else:
                disc_loss = dloss_fun(crit_fake_pred, crit_real_pred)

            disc_loss.backward()
            if args.loss_fun == 'wgan':
                torch.nn.utils.clip_grad_norm_(disc.parameters(), 0.01)
            disc_opt.step()

            gen_opt.zero_grad()
            fake_noise_ = get_noise(args.seq_len, feature_dim, device=args.device)
            fake_ = gen(fake_noise_).detach()
            crit_fake_pred_ = disc(fake_)
            gen_loss = gloss_fun(crit_fake_pred_)
            gen_loss.backward()
            gen_opt.step()
        
        if epoch % 10 == 0:
            print('Generator Loss: {:.2f} Discrim Loss: {:.2f}'.format(gen_loss, disc_loss))
            plt.style.use('fivethirtyeight')
            plt.figure(figsize=(20,7), dpi=300)
            plt.rcParams["font.size"] = 18
            ax = plt.subplot(131)
            ax.plot(fake_noise_[0, :].cpu().detach().numpy())
            ax.set_title('Noise')
            ax = plt.subplot(132)
            ax.plot(fake_[0, :].cpu().detach().numpy())
            ax.set_title('fake')
            ax = plt.subplot(133)
            ax.plot(real[0, :,].cpu().detach().numpy())
            ax.set_title('Real')
            plt.show()

            
        g_losses.append(gen_loss.detach().cpu().numpy())
        d_losses.append(disc_loss.detach().cpu().numpy())

    torch.save(gen, '../Results/9nov/' + args.model_name + '_' + args.loss_fun + '.pkl')

    np.savetxt('../Results/9nov/G_' +  args.model_name + '_' + args.loss_fun + '.csv',
               np.asarray(g_losses), delimiter=',')
    np.savetxt('../Results/9nov/D_'  +  args.model_name + '_'  + args.loss_fun + '.csv',
               np.asarray(d_losses), delimiter=',')


if __name__ == '__main__':
    parser_ = argparse.ArgumentParser()

    parser_.add_argument('--data_name',
                         choices=['stock', 'energy', 'sine', 'all_intel'],
                         default='all_intel', type=str)
    parser_.add_argument('--seq_len', default=168, type=int)
    parser_.add_argument('--n_epochs', default=50, type=int)
    parser_.add_argument('--batch_size', default=10, type=int)
    parser_.add_argument('--learning_rate', default=1e-6, type=float)
    parser_.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', type=str)
    parser_.add_argument('--model_name', default='1667827583', type=str)
    parser_.add_argument('--scaler_name', default='MinMax', type=str)
    parser_.add_argument('--loss_fun', default='gan', choices=['gan', 'lsgan',
                                                                 'wgan', 'wgan_gp', 'dragan'],
                         type=str)
    args_ = parser_.parse_args()
    main(args_)

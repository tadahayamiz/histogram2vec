# -*- coding: utf-8 -*-
"""

not CLI package

@author: tadahaya
"""
# packages installed in the current environment
import os
import datetime
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import glob

# original packages in src
from .src import utils
from .src import data_handler as dh
from .src.models import VAE
from .src.models import loss_function

SEP = os.sep
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Hist2vec:
    def __init__(
            self, workdir:str="", datafile:str="", seed:int=222,
            num_step:int=100000, batch_size:int=128, lr:float=1e-4,
            n_monitor:int=1000, encoder_output_size=16, dim_latent=64
            ):
        self.workdir = workdir
        self.seed = 222
        self.num_step = num_step
        self.batch_size = batch_size
        self.lr = lr
        self.enc_out = encoder_output_size
        self.dim_latent = dim_latent
        self.n_monitor = n_monitor
        utils.fix_seed(seed=seed, fix_gpu=False) # for seed control
        self._now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.dir_name = self.workdir + SEP + 'results' + SEP + self._now # for output
        self.datafile = datafile
        if not os.path.exists(self.datafile):
            tmp = sorted(glob.glob(self.workdir + SEP + '*.npz'), reverse=True)[0]
            if len(tmp) > 0:
                self.datafile = tmp
                print(f"load the following data: {tmp}")
            else:
                raise ValueError("!! Give data path !!")
        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)
        self.logger = utils.init_logger(
            __name__, self.dir_name, self._now, level_console='debug'
            )
        self.model = None


    def prepare_data(self):
        """
        データの読み込み・ローダーの準備を実施
        加工済みのものをdataにおいておくか, argumentで指定したパスから呼び出すなりしてデータを読み込む
        inference用を読み込む際のものも用意しておくと楽
        
        """
        # train_trans = transforms.Compose([
        #     transforms.ToTensor(),
        # ])
        # test_trans = transforms.Compose([
        #     transforms.ToTensor(),
        # ])
        dataset = np.load(self.datafile)
        idx = int(dataset["input"].shape[0] * 0.9)
        input = torch.tensor(dataset["input"]).float()
        output = torch.tensor(dataset["output"]).float()
        train_loader, test_loader = dh.prep_data(
            input[:idx], output[:idx], input[idx:], output[idx:],
            batch_size=self.batch_size, transform=(None, None)
            )
        # train_loader, test_loader = dh.prep_data(
        #     input[:idx], output[:idx], input[idx:], output[idx:],
        #     batch_size=self.batch_size, transform=(train_trans, test_trans)
        #     )
        return train_loader, test_loader


    def prepare_model(self):
        """
        model, loss, optimizer, schedulerの準備

        """
        model = VAE(
            color_channels=1, pooling_kernels=(2, 2),
            encoder_output_size=self.enc_out, dim_latent=self.dim_latent
            )
        model.to(DEVICE)
        criterion = loss_function
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        return model, criterion, optimizer


    def train_step(self, model, train_loader, test_loader, criterion, optimizer):
        """
        epoch単位の学習構成, なくとも良い
        
        """
        model.train() # training
        train_batch_loss = []
        train_batch_rl = []
        train_batch_kld = []
        for data_in, data_out in train_loader:
            data_in, data_out = data_in.to(DEVICE), data_out.to(DEVICE) # put data on GPU
            optimizer.zero_grad() # reset gradients
            output, mu, logvar = model(data_in) # forward
            loss, rl, kld = criterion(output, data_out, mu, logvar) # calculate loss
            loss.backward() # backpropagation
            optimizer.step() # update parameters
            train_batch_loss.append(loss.item())
            train_batch_rl.append(rl.item())
            train_batch_kld.append(kld.item())
        model.eval() # test (validation)
        test_batch_loss = []
        test_batch_rl = []
        test_batch_kld = []
        with torch.no_grad():
            for data_in, data_out in test_loader:
                data_in, data_out = data_in.to(DEVICE), data_out.to(DEVICE)
                output, mu, logvar = model(data_in)
                loss, rl, kld = criterion(output, data_out, mu, logvar)
                test_batch_loss.append(loss.item())
                test_batch_rl.append(rl.item())
                test_batch_kld.append(kld.item())
        train_loss = (
            np.mean(train_batch_loss), np.mean(train_batch_rl), np.mean(train_batch_kld)
            )
        test_loss = (
            np.mean(test_batch_loss), np.mean(test_batch_rl), np.mean(test_batch_kld)
            )
        return model, train_loss, test_loss


    def fit(self, model, train_loader, test_loader, criterion, optimizer):
        """
        学習
        model, train_loss, test_loss (valid_loss)を返す
        
        """
        train_loss = []
        train_rl = []
        train_kld = []
        test_loss = []
        test_rl = []
        test_kld = []
        for step in trange(self.num_step):
            model, train_step_loss, test_step_loss = self.train_step(
                model, train_loader, test_loader, criterion, optimizer
                )
            train_loss.append(train_step_loss[0])
            train_rl.append(train_step_loss[1])
            train_kld.append(train_step_loss[2])
            test_loss.append(test_step_loss[0])
            test_rl.append(test_step_loss[1])
            test_kld.append(test_step_loss[2])
            if step % self.n_monitor == 0:
                self.logger.info(
                    f'step: {step} // train_loss: {train_step_loss[0]:.4f} // valid_loss: {test_step_loss[0]:.4f}'
                    )
        return model, (train_loss, train_rl, train_kld), (test_loss, test_rl, test_kld)


    def main(self):
        start = time.time() # for time stamp
        # 1. data prep
        train_loader, test_loader = self.prepare_data()
        self.logger.info(
            f'num_training_data: {len(train_loader)}, num_test_data: {len(test_loader)}'
            )
        # 2. model prep
        model, criterion, optimizer = self.prepare_model()
        # 3. training
        self.model, train_loss, test_loss = self.fit(
            model, train_loader, test_loader, criterion, optimizer
            )
        utils.plot_progress(train_loss, test_loss, self.num_step, self.dir_name)
        utils.summarize_model(model, next(iter(train_loader))[0], self.dir_name)
        # 4. save results & config
        utils.to_logger(self.logger, name='loss', obj=criterion)
        utils.to_logger(
            self.logger, name='optimizer', obj=optimizer, skip_keys={'state', 'param_groups'}
            )
        self.logger.info('elapsed_time: {:.2f} min'.format((time.time() - start)/60))
        torch.save(model.state_dict(), self.dir_name + SEP + 'model_weight.pth')


    def load_model(self, url):
        self.model, _, _ = self.prepare_model()
        self.model.load_state_dict(torch.load(url))


    def encode(self, x):
        if self.model is None:
            raise ValueError("!! No trained model !!")
        with torch.no_grad():
            return self.model.encode(x)
        
    

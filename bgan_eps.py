from __future__ import division
import numpy as np
import copy
import os,sys
import time
import argparse
import importlib
import random
import model
import util
import torch
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import sem
torch.manual_seed(0)

class BGAN(object):
    def __init__(self, g_net, h_net, dx_net, dy_net, x_sampler, y_sampler, batch_size, learning_rate, alpha, beta, filename, vib, scaler, device, top_file, ensemble):
        self.device = device
        self.g_net = g_net.to(self.device)
        self.h_net = h_net.to(self.device)
        self.dx_net = dx_net.to(self.device)
        self.dy_net = dy_net.to(self.device)
        self.x_sampler = x_sampler
        self.y_sampler = y_sampler
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.x_dim = self.dx_net.input_dim
        self.y_dim = self.dy_net.input_dim
        self.lr = learning_rate
        self.top_file = top_file
        self.g_h_optim = torch.optim.Adam(list(self.g_net.parameters()) + list(self.h_net.parameters()), lr = self.lr, betas = (0.5, 0.999))
        self.d_optim = torch.optim.Adam(list(self.dx_net.parameters()) + list(self.dy_net.parameters()), lr = self.lr, betas = (0.5, 0.999))
        self.bgan_data = []
        self.vib = vib
        self.vibmax = [max(idx) for idx in zip(*self.vib)]
        self.vibmin = [min(idx) for idx in zip(*self.vib)]
        self.scaler = scaler

        self.reac_coord = np.linspace(self.vibmax[0],self.vibmin[0],ensemble+1)

    def generator_loss(self, x, y, epoch, cv_epoch):
        y_ = self.g_net(x)
        x_ = self.h_net(y)

        y__ = self.g_net(x_)
        x__ = self.h_net(y_)

        if epoch >= cv_epoch:
            self.generate_data(y__)

        dy_ = self.dy_net(y_)
        dx_ = self.dx_net(x_)

        l1_loss_x = torch.mean(torch.abs(x - x__))
        l1_loss_y = torch.mean(torch.abs(y - y__))

        l2_loss_x = torch.mean((x - x__)**2)
        l2_loss_y = torch.mean((y - y__)**2)
        
        g_loss_adv = torch.mean((torch.ones_like(dy_) - dy_)**2)
        h_loss_adv = torch.mean((torch.ones_like(dx_) - dx_)**2)
        
        g_loss = g_loss_adv + self.alpha * l2_loss_x + self.beta * l2_loss_y
        h_loss = h_loss_adv + self.alpha * l2_loss_x + self.beta * l2_loss_y
        g_h_loss = g_loss_adv + h_loss_adv + self.alpha * l2_loss_x + self.beta * l2_loss_y
        return g_loss, h_loss, g_h_loss
        
    def discriminator_loss(self, x, y):
        fake_x = self.h_net(y)
        fake_y = self.g_net(x)

        dx = self.dx_net(x)
        dy = self.dy_net(y)

        d_fake_x = self.dx_net(fake_x)
        d_fake_y = self.dy_net(fake_y)
        
        dx_loss = (torch.mean((torch.ones_like(dx) - dx)**2) + torch.mean((torch.zeros_like(d_fake_x) - d_fake_x)**2))/2.0
        dy_loss = (torch.mean((torch.ones_like(dy) - dy)**2) + torch.mean((torch.zeros_like(d_fake_y) - d_fake_y)**2))/2.0

        d_loss = dx_loss + dy_loss
        return dx_loss, dy_loss, d_loss

    def generate_data(self, noise):
        dummy = noise.detach().cpu().numpy()
        for items in dummy:
            self.bgan_data.append(list(items))

    def undo_norm(self, inp):
        X = inp.copy()
        inversed = self.scaler.inverse_transform(X)
        output = inversed.T
        for i, items in enumerate(output):
            for j, item in enumerate(items):
                if item > self.vibmax[i]:
                    output[i][j] = item - (self.vibmax[i]-self.vibmin[i])
        return inversed

    def topology(self, X):
        X = np.delete(X,0,1)
        return util.EC_classifier(X,50,self.top_file).get_dof()

    def calc(self, X, classifier):
        conv_fact = -1.987204259e-3*298.15 #kcal/mol
        out = []
        for i in range(0,len(self.reac_coord)-1):
            eva = []
            for items in X:
                if items[0] >= self.reac_coord[i+1] and items[0] < self.reac_coord[i]:
                    eva.append(items)
            if len(eva) == 0: out.append(None)
            else:
                eva = np.delete(eva,0,1)
                dummy = []
                for item in classifier:
                    dummy.append([row[item] for row in eva])
                dummy = np.array(dummy).T
                MIST = util.EC(dummy,50,self.top_file).output()
                out.append(MIST*conv_fact)
        print([(self.reac_coord[j+1]+self.reac_coord[j])/2 for j in range(0,len(self.reac_coord)-1)])
        return list(out)
    
    def train(self, epochs, cv_epoch):
        data_y_train = copy.copy(self.y_sampler.X_train)
        start_time = time.time()
        for epoch in range(epochs):
            np.random.shuffle(data_y_train)
            batch_idxs = len(data_y_train) // self.batch_size
            for idx in range(batch_idxs):
                bx = self.x_sampler.get_batch(self.batch_size)
                by = data_y_train[self.batch_size*idx:self.batch_size*(idx+1)]
                x = torch.Tensor(bx).to(self.device)
                y = torch.Tensor(by).to(self.device)

                self.g_h_optim.zero_grad()
                g_loss, h_loss, g_h_loss = self.generator_loss(x, y, epoch, cv_epoch)
                g_h_loss.backward()
                self.g_h_optim.step()

                self.d_optim.zero_grad()
                dx_loss, dy_loss, d_loss = self.discriminator_loss(x, y)
                d_loss.backward()
                self.d_optim.step()

            print('Epoch [%d] Time [%5.4f] g_loss [%.4f] h_loss [%.4f] g_h_loss [%.4f] dx_loss [%.4f] dy_loss [%.4f] d_loss [%.4f]' % (epoch, time.time() - start_time, g_loss, h_loss, g_h_loss, dx_loss, dy_loss, d_loss))
            if epoch+1 == epochs:
                reform = self.undo_norm(self.bgan_data)
                classifier = self.topology(reform)
                print(self.calc(reform, classifier))
                sys.exit()

def reshape(inp, lst):
    X = inp.copy().T
    Xmin = [min(items) for items in X]
    Xmax = [max(items) for items in X]
    for num in lst:
        for i, val in enumerate(X[num]):
            if X[num][i] <= 0:
                X[num][i] = val + (Xmax[num] - Xmin[num])
    return X.T

def gauss_check(X):
    num = []
    for i in range(len(X[0])):
        dof = [row[i] for row in X]
        sep = False
        if max(dof) >= 3.1 and min(dof) <= 3.1:
            sep = True
        if sep == True:
            num.append(i)
    return num

def norm_vib(inp):
    X = inp.copy()
    output = []
    num = gauss_check(X)
    X_trans = reshape(X,num)
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_trans)
    scaled = scaler.transform(X_trans)
    return scaler, scaled

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--lr',type=float, default=0.00015,help='BGAN learning rate')
    parser.add_argument('--bs', type=int, default=64,help='batch size for training')
    parser.add_argument('--epochs', type=int, default=130,help='maximum training epoches')
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--beta', type=float, default=10.0)
    parser.add_argument('--filename', type=str, default='./temporary/trajectory.npy', help='file name')
    parser.add_argument('--topology', type=str, default='./temporary/topology.txt', help='topology file')
    parser.add_argument('--nb', type=int, default=32, help='hidden node')
    parser.add_argument('--ensemble', type=int, default=10, help='number of structural ensembles')

    args = parser.parse_args()
    learning_rate = args.lr
    batch_size = args.bs
    epochs = args.epochs
    cv_epoch = 1
    alpha = args.alpha
    beta = args.beta
    filename = args.filename
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    top_file = args.topology
    nb = args.nb
    ensemble = args.ensemble
    
    vib = np.load(filename)
    scaler, scaledvib = norm_vib(vib)
    ys = util.BAT_sampler(scaledvib)
    dim = len(vib[0])

    xs = util.Gaussian_sampler(mean=np.zeros(dim),sd=0.5)
    #xs = util.Uniform_sampler(mean=np.zeros(dim))

    g_net = model.Generator(input_dim=dim, output_dim = dim, nb_units = nb)
    h_net = model.Generator(input_dim=dim, output_dim = dim, nb_units = nb)
    dx_net = model.Discriminator(input_dim=dim, nb_units = nb)
    dy_net = model.Discriminator(input_dim=dim, nb_units = nb)

    BGANEPS = BGAN(g_net, h_net, dx_net, dy_net, xs, ys, batch_size, learning_rate, alpha, beta, filename, vib, scaler, device, top_file, ensemble)
    BGANEPS.train(epochs=epochs,cv_epoch=cv_epoch)

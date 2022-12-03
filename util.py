from __future__ import division
import numpy as np
import os,sys
import math
import scipy
import argparse
import importlib
import random
from scipy.stats import norm

class BAT_sampler(object):
    def __init__(self, vib):
        self.X_train = self.shuffle(vib)
        self.Y = None
        self.nb_train = self.X_train.shape[0]
        self.mean = 0
        self.sd = 0
    def shuffle(self,data):
        rng = np.random.RandomState(42)
        rng.shuffle(data)
        return data
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.nb_train, size = batch_size)
        if label:
            return self.X_train[indx, :], self.Y[indx]
        else:
            return self.X_train[indx, :]
    def load_all(self):
        return self.X_train, None

class Gaussian_sampler(object):
    def __init__(self, mean, sd=1):
        self.mean = mean
        self.sd = sd
        np.random.seed(1024)
    def get_batch(self,batch_size):
        return np.random.normal(self.mean, self.sd, (batch_size,len(self.mean)))

class Uniform_sampler(object):
    def __init__(self, mean):
        self.mean = mean
        self.dim = dim 
        np.random.seed(1024)
    def get_batch(self, batch_size):
        return np.random.uniform(self.mean-0.5,self.mean+0.5,(batch_size,len(self.mean)))

class Topology:
    def __init__(self, top_input_file):
        self.bond, self.angle, self.torsion = 0, 0, 0
        with open(top_input_file, 'r') as f:
            for line in f:
                if len(line.split()) == 1:
                    self.atom_num = eval(line)
                if len(line.split()) == 2:
                    self.bond += 1
                if len(line.split()) == 3:
                    self.angle += 1
                if len(line.split()) == 4:
                    self.torsion += 1

class EC_classifier:
    def __init__(self, vib, bin_size, top_file):
        self.vib = vib
        self.bin_size = bin_size
        self.sample_size = len(self.vib)
        self.dof = len(self.vib[0])
        self.list1D = []
        self.dof_list = []
        self.topology = Topology(top_file)
        self.atom_num = self.topology.atom_num
        for i in range (0,self.topology.bond):
            self.dof_list.append('b')
        for i in range (0,self.topology.angle):
            self.dof_list.append('a')
        for i in range (0,self.topology.torsion):
            self.dof_list.append('d')
        if self.dof != len(self.dof_list):
            print("Wrong number of DoF!")
            
        for i in range (0, self.dof):
            val = self.entropy_1D([row[i] for row in self.vib], self.dof_list[i])
            self.list1D.append(val)

    def get_dof(self):
        x1 = np.argsort(self.list1D[:self.topology.bond])[::-1][:self.atom_num-1]
        x2 = np.add(np.argsort(self.list1D[self.topology.bond:self.topology.bond+self.topology.angle])[::-1][:self.atom_num-2],self.topology.bond)
        x3 = np.add(np.argsort(self.list1D[self.topology.bond+self.topology.angle:])[::-1][:self.atom_num-3],self.topology.bond+self.topology.angle)
        x = np.concatenate((x1,x2,x3))
        x.sort()
        return x

    def get_jacobian(self, X_bin_edge, xjtype):
        if xjtype == 'b':
            return X_bin_edge**2
        elif xjtype == 'd':
            return 1
        else:
            return math.sin(X_bin_edge)

    def entropy_1D(self, X, jtype):
        occupied_bin = 0
        counts, bin_edges = np.histogram(X, self.bin_size)
        prob_den = counts/len(X)
        dx = (np.max(X)-np.min(X))/self.bin_size
        bin_edges += dx/2
        entropy1D = 0
        for i in range(0, len(prob_den)):
            if (prob_den[i] != 0):
                entropy1D += prob_den[i] * math.log(prob_den[i]/(self.get_jacobian(bin_edges[i],jtype)*dx))
                occupied_bin += 1
        entropy1D = -entropy1D + (occupied_bin-1) / (2 * self.sample_size)
        return entropy1D

class EC:
    def __init__(self, vib, bin_size, top_file):
        self.vib = vib
        self.bin_size = bin_size
        self.sample_size = len(self.vib)
        self.dof = len(self.vib[0])
        self.list1D, self.list2D, self.MIlist, self.MISTlist = [], [], [], []
        self.entropy, self.MI, self.MIST = 0, 0, 0
        self.topology = Topology(top_file)
        self.atom_num = self.topology.atom_num
        self.dof_list = []
        for i in range (0,self.atom_num-1):
            self.dof_list.append('b')
        for i in range (0,self.atom_num-2):
            self.dof_list.append('a')
        for i in range (0,self.atom_num-3):
            self.dof_list.append('d')

        for i in range (0, self.dof):
            val = self.entropy_1D([row[i] for row in self.vib], self.dof_list[i])
            self.list1D.append(val)
            self.entropy+=val

        for i in range (0, self.dof):
            dummy = []
            for j in range(i+1, self.dof):
                self.list2D.append(self.entropy_2D([row[i] for row in self.vib], self.dof_list[i], [row[j] for row in self.vib], self.dof_list[j]))
                dummy.append(self.list1D[i] + self.list1D[j] - self.list2D[-1])
                if j == self.dof-1: break
            self.MI+=np.sum(dummy)
            self.MIlist.append(dummy)
        self.MIlist.pop()
        for items in self.MIlist:
            self.MIST += np.max(items)
        for items in self.MIlist:
            self.MISTlist.append([np.max(items),items.index(np.max(items))])

    def output(self):
        return self.entropy-self.MIST

    def get_jacobian(self, X_bin_edge, xjtype, Y_bin_edge, yjtype):
        if Y_bin_edge == None and yjtype == None:
            if xjtype == 'b':
                return X_bin_edge**2
            elif xjtype == 'd':
                return 1
            else:
                return math.sin(X_bin_edge)
        else:
            if xjtype == 'b' and yjtype == 'b':
                return X_bin_edge**2 * Y_bin_edge**2
            elif xjtype == 'b' and yjtype == 'a':
                return X_bin_edge**2 * math.sin(Y_bin_edge)
            elif xjtype == 'b' and yjtype == 'd':
                return X_bin_edge**2
            elif xjtype == 'a' and yjtype == 'a':
                return math.sin(X_bin_edge)*math.sin(Y_bin_edge)
            elif xjtype == 'a' and yjtype == 'd':
                return math.sin(X_bin_edge)
            else:
                return 1         
        
    def entropy_1D(self, X, jtype):
        occupied_bin = 0
        counts, bin_edges = np.histogram(X, self.bin_size)
        prob_den = counts/len(X)
        dx = (np.max(X)-np.min(X))/self.bin_size
        bin_edges += dx/2
        entropy1D = 0
        for i in range(0, len(prob_den)):
            if (prob_den[i] != 0):
                entropy1D += prob_den[i] * math.log(prob_den[i]/(self.get_jacobian(bin_edges[i],jtype, None, None)*dx))
                occupied_bin += 1
        entropy1D = -entropy1D + (occupied_bin-1) / (2 * self.sample_size)
        return entropy1D
    
    def entropy_2D(self, X, xjtype, Y, yjtype):
        occupied_bin = 0
        H, X_bin_edges, Y_bin_edges = np.histogram2d(X, Y, bins=self.bin_size)
        dx, dy = (np.max(X_bin_edges)-np.min(X_bin_edges))/self.bin_size , (np.max(Y_bin_edges)-np.min(Y_bin_edges))/self.bin_size
        X_bin_edges += (dx/2)
        Y_bin_edges += (dy/2)
        H = H.T / self.sample_size
        entropy2D = 0
        for row in range(0, self.bin_size):
            for col in range(0, self.bin_size):
                if (H[row][col] != 0):
                    entropy2D += H[row][col] * math.log(H[row][col] / (self.get_jacobian(X_bin_edges[col],xjtype,Y_bin_edges[row],yjtype) * dx * dy))
                    occupied_bin += 1
        entropy2D = -entropy2D + (occupied_bin-1) / (2 * self.sample_size)
        return entropy2D


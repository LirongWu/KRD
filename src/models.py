import copy
import numpy as np
from scipy.stats import entropy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, SAGEConv, GATConv
from utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout_ratio, norm_type='none'):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, feats):
        h = feats
        h_list = []

        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != self.num_layers - 1:
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)
            h_list.append(h)

        return h_list, h


class GCN(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout_ratio, activation):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(GraphConv(input_dim, output_dim, activation=activation))
        else:
            self.layers.append(GraphConv(input_dim, hidden_dim, activation=activation))
            for i in range(num_layers - 2):
                self.layers.append(GraphConv(hidden_dim, hidden_dim, activation=activation))
            self.layers.append(GraphConv(hidden_dim, output_dim))

    def forward(self, g, feats):
        h = feats
        h_list = []

        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h = self.dropout(h)
            h_list.append(h)
            
        return h_list, h

class GAT(nn.Module):
    def __init__(
        self,num_layers, input_dim, hidden_dim, output_dim, dropout_ratio, activation, num_heads=4, attn_drop=0.3, negative_slope=0.2, residual=False):
        super(GAT, self).__init__()
        hidden_dim //= num_heads
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        heads = ([num_heads] * num_layers) + [1]

        self.layers.append(GATConv(input_dim, hidden_dim, heads[0], dropout_ratio, attn_drop, negative_slope, False, activation))
        for l in range(1, num_layers - 1):
            self.layers.append(GATConv(hidden_dim * heads[l-1], hidden_dim, heads[l], dropout_ratio, attn_drop, negative_slope, residual, activation))
        self.layers.append(GATConv(hidden_dim * heads[-2], output_dim, heads[-1], dropout_ratio, attn_drop, negative_slope, residual, None))

    def forward(self, g, feats):
        h = feats
        h_list = []

        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h = h.flatten(1)
            else:
                h = h.mean(1)
            h_list.append(h)

        return h_list, h


class GraphSAGE(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout_ratio, activation):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.activation = activation

        if num_layers == 1:
            self.layers.append(SAGEConv(input_dim, output_dim, aggregator_type='gcn'))
        else:
            self.layers.append(SAGEConv(input_dim, hidden_dim, aggregator_type='gcn'))
            for i in range(num_layers - 2):
                self.layers.append(SAGEConv(hidden_dim, hidden_dim, aggregator_type='gcn'))
            self.layers.append(SAGEConv(hidden_dim, output_dim, aggregator_type='gcn'))

    def forward(self, g, feats):
        h = feats
        h_list = []

        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h = self.activation(h)
                h = self.dropout(h)
            h_list.append(h)
            
        return h_list, h


class Model(nn.Module):
    def __init__(self, param, model_type=None):
        super(Model, self).__init__()

        if model_type == 'teacher':
            self.model_name = param["teacher"]
        else:
            self.model_name = param["student"]
        
        if "MLP" in self.model_name:
            self.encoder = MLP(
                num_layers=param["num_layers"],
                input_dim=param["feat_dim"],
                hidden_dim=param["hidden_dim"],
                output_dim=param["label_dim"],
                dropout_ratio=param["dropout_s"],
                norm_type=param["norm_type"],
            )
        elif "GCN" in self.model_name:
            self.encoder = GCN(
                num_layers=param["num_layers"],
                input_dim=param["feat_dim"],
                hidden_dim=param["hidden_dim"],
                output_dim=param["label_dim"],
                dropout_ratio=param["dropout_t"],
                activation=F.relu,
            )
        elif "GAT" in self.model_name:
            self.encoder = GAT(
                num_layers=param["num_layers"],
                input_dim=param["feat_dim"],
                hidden_dim=param["hidden_dim"],
                output_dim=param["label_dim"],
                dropout_ratio=param["dropout_t"],
                activation=F.relu,
            )
        elif "SAGE" in self.model_name:
            self.encoder = GraphSAGE(
                num_layers=param["num_layers"],
                input_dim=param["feat_dim"],
                hidden_dim=param["hidden_dim"],
                output_dim=param["label_dim"],
                dropout_ratio=param["dropout_t"],
                activation=F.relu,
            )

    def forward(self, g, feats):
        if "MLP" in self.model_name:
            return self.encoder(feats)[1]
        else:
            return self.encoder(g, feats)[1]


class Com_KD_Prob(nn.Module):
    def __init__(self, param):
        super(Com_KD_Prob, self).__init__()
        self.param = param

        self.power = param['init_power']
        self.momentum = param['momentum']
        self.bins_num = param['bins_num']
        self.noise_level = 1.0

        self.delta_entropy = None

    def initialization(self, model, g, feats):

        g = g.to(device)

        model.eval()
        data_teacher = model.forward(g, feats).softmax(dim=-1).detach().cpu().numpy()
    
        weight_t = []
        for i in range(data_teacher.shape[0]):
            weight_t.append(entropy(data_teacher[i]))
        weight_t = np.array(weight_t)

        feats_noise = copy.deepcopy(feats)
        feats_noise += torch.randn(feats.shape[0], feats.shape[1]).to(device) * self.noise_level
        out_noise = model.forward(g, feats_noise).softmax(dim=-1).detach().cpu().numpy()

        weight_s = np.zeros(feats.shape[0])
        for i in range(feats.shape[0]):
            weight_s[i] = np.abs(entropy(out_noise[i]) - weight_t[i])
        self.delta_entropy = weight_s / np.max(weight_s)

    def func(self, x, a):
        return 1 - x ** a

    def fit(self, func, xdata, ydata):
        popt, pcov = curve_fit(func, xdata, ydata, p0 = [self.power])
        return popt[0]

    def updata(self, logits_t, logits_s):

        weight_true = []
        weight_false = []
        for i in range(logits_t.shape[0]):
            if np.argmax(logits_t[i]) == np.argmax(logits_s[i]):
                weight_true.append(self.delta_entropy[i])
            else:
                weight_false.append(self.delta_entropy[i])
        weight_true = np.array(weight_true)
        weight_false = np.array(weight_false)

        hist_t, bins_t = np.histogram(weight_true, bins=self.bins_num, range=(0, 1))
        hist_s, bins_s = np.histogram(weight_false, bins=self.bins_num, range=(0, 1))

        prob = np.zeros(hist_t.shape[0])
        for i in range(hist_t.shape[0]):
            prob[i] = 1.0 * hist_t[i] / (hist_t[i] + hist_s[i] + 1e-6)
        prob = (prob - np.min(prob)) / (np.max(prob) - np.min(prob))
        update_power = self.fit(self.func, bins_t[:-1] + 0.05, prob)
        self.power = self.momentum * self.power + (1-self.momentum) * update_power

    def predict_prob(self):
        return 1 - self.delta_entropy ** self.power


    def plot_fit_curve(self, logits_t, logits_s, labels):

        weight_true = []
        weight_false = []
        for i in range(logits_t.shape[0]):
            if np.argmax(logits_t[i]) == labels[i] and np.argmax(logits_s[i]) == labels[i]:
                weight_true.append(self.delta_entropy[i])
            else:
                weight_false.append(self.delta_entropy[i])
        weight_true = np.array(weight_true)
        weight_false = np.array(weight_false)

        hist_t, bins_t = np.histogram(weight_true, bins=self.bins_num, range=(0, 1))
        hist_s, bins_s = np.histogram(weight_false, bins=self.bins_num, range=(0, 1))

        prob = np.zeros(hist_t.shape[0])
        for i in range(hist_t.shape[0]):
            prob[i] = 1.0 * hist_t[i] / (hist_t[i] + hist_s[i] + 1e-6)
        prob = (prob - np.min(prob)) / (np.max(prob) - np.min(prob))
    
        plt.bar(bins_t[:-1] + 0.05, prob, color = 'r', edgecolor = 'black', alpha = 0.3, width=0.7/self.bins_num)

        x = np.arange(0, 1, 0.001)
        y = 1 - x ** self.power

        plt.plot(x, y)
        plt.grid()

        check_writable('../outputs/images/{}/'.format(self.param['dataset']), overwrite=False)
        plt.savefig('../outputs/images/{}/drowning_{}_{}_{}.png'.format(self.param['dataset'], self.param['init_power'], self.param['momentum'], self.param['seed']), dpi=200)
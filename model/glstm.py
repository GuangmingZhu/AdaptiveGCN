import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, num_subset=3):
        super(unit_gcn, self).__init__()
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        self.A = torch.tensor(A, dtype=torch.float32, requires_grad=False)
        self.num_subset = num_subset

        self.conv = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.res = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv[i], self.num_subset)

    def forward(self, x):
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            f = self.conv[i](x)
            N, C, T, V = f.size()
            z = torch.matmul(f.view(N, C * T, V), A[i]).view(N, C, T, V)
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.res(x)
        return self.relu(y)


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class GraphTempLSTM_unit(nn.Module):
    def __init__(self, isize, hsize, layers=(1,1), bidirectional=(False,False)):
        super(GraphTempLSTM_unit, self).__init__()
        self.isize = isize
        self.hsize = hsize
        self.glstm = nn.LSTM(isize, hsize, layers[0], bidirectional=bidirectional[0])
        self.tlstm = nn.LSTM(hsize, hsize, layers[1], bidirectional=bidirectional[1])
        if bidirectional[0]:
            self.glinear = nn.Linear(hsize*4, hsize)
        else:
            self.glinear = nn.Linear(hsize*2, hsize)
        if bidirectional[1]:
            self.tlinear = nn.Linear(hsize*4, hsize)
        else:
            self.tlinear = nn.Linear(hsize*2, hsize)
        nn.init.normal_(self.glinear.weight, 0, math.sqrt(2. / hsize))
        nn.init.normal_(self.tlinear.weight, 0, math.sqrt(2. / hsize))

    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(3, 2, 0, 1).contiguous().view(V, T * N, C)
        #x = x[torch.randperm(x.size(0))]

        y,_ = self.glstm(x)
        ymean = torch.mean(y, 0)
        ymax,_ = torch.max(y, 0)
        ynew = torch.cat((ymean, ymax), dim=1)
        ynew = self.glinear(ynew)

        z,_ = self.tlstm(ynew.view(T, N, self.hsize))
        zmean = torch.mean(z, 0)
        zmax,_ = torch.max(z, 0)
        znew = torch.cat((zmean, zmax), dim=1)
        znew = self.tlinear(znew)
        
        return znew


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3, glstm_layers=(2,2),glstm_bidirs=(True,True)):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l5 = TCN_GCN_unit(128, 128, A)
        self.l6 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l7 = TCN_GCN_unit(256, 256, A)

        self.s1 = GraphTempLSTM_unit(256, 256, glstm_layers, glstm_bidirs)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.s1(x) 

        # T,N*M,C
        c_new = x.size(1)
        x = x.view(N, M, c_new)
        x = x.mean(1)

        return self.fc(x)

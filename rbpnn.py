import torch_rbf as rbf
import torch
import torch.nn as nn
from torch.nn import Linear


class RBPNN(nn.Module):

    """
    batch_size: how many samples the module process in one batch
    out_types: how many types do sample set has
    basis_func
    """

    def __init__(self, batch_size, features, out_types, basis_func=rbf.gaussian, log_sigma=0, lin=True, wrbf=False, wsum=False):
        super(RBPNN, self).__init__()
        self.batch_size = batch_size
        self.features = features
        self.out_types = out_types
        self.basis_func = basis_func
        self.log_sigma = log_sigma
        self.rbfs = nn.ModuleList()
        self.rbf_num = 0

        # recording the cluster info for each node in rbfs. (r, n)
        # r: index for rbfs,
        # n: index for cluster, lenth: num of node, value: range(num of clusters)
        self.cluster_info = []

        self.lin = lin
        if self.lin:
            self.lin1 = Linear(self.out_types, self.out_types)

        self.wrbf = wrbf
        self.wsum = wsum
        if self.wsum:
            self.weight_sum = nn.Parameter(torch.Tensor(out_types))

    def reset_parameters(self):
        if self.wsum:
            nn.init.normal_(self.weight_sum)

        return

    """
    input: Tensor.shape(batch_size, features)
    """
    def forward(self, input):
        tensorlist = []
        for rbf in self.rbfs:
            tensorlist.append(rbf(input))
        rbf_sum_layer = self.sum_rbf_out(tensorlist, input.size(0))
        if self.wsum:
            raise NotImplementedError
        if self.lin:
            return self.lin1(rbf_sum_layer)
        return rbf_sum_layer

    def add_rbf(self, clusters: list, new_centres):
        self.cluster_info.append(clusters)
        new_rbf = rbf.RBF(self.features, len(clusters), self.basis_func, new_centres, self.log_sigma, self.wrbf)
        self.rbfs.append(new_rbf)

    def sum_rbf_out(self, tensorlist, input_len):
        sum = torch.zeros(self.out_types, input_len)
        for i, nodes in enumerate(self.cluster_info):
            for j, node in enumerate(nodes):
                sum[node] += tensorlist[i].t()[j]
        return sum.t()

    def ban_centres(self, centres_info: list):
        raise NotImplementedError

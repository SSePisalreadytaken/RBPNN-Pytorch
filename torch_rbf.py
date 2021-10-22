import torch
import torch.nn as nn

# RBF Layer

class RBF(nn.Module):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})

    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample

    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size

    Attributes:
        centres: the learnable centres of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.
        
        log_sigmas: logarithm of the learnable scaling factors of shape (out_features).
        
        basis_func: the radial basis function used to transform the scaled
            distances.
    """

    def __init__(self, in_features, out_features, basis_func, centres=torch.Tensor(), log_sigmas=0, wout=False):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.centres_in = centres
        self.log_sigmas = nn.Parameter(torch.Tensor(out_features))
        self.log_sigmas_in = log_sigmas
        self.basis_func = basis_func
        self.wout = wout
        if self.wout:
            self.weight_out = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        if self.centres_in.size() == self.centres.size():
            self.centres=self.centres_in
        else:
            nn.init.normal_(self.centres, 0, 1)
            print("normalized centres")

        if self.wout:
            nn.init.normal_(self.weight_out)

        nn.init.constant_(self.log_sigmas, self.log_sigmas_in)

    def read_centres(self):
        return self.centres

    """

    # pos array, centre tensor
    def add_centres(self, pos, centres):
        new_centres = insert_tensor(self.centres, pos, centres)
        log_sigmas = torch.ones(new_centres.size(0) - self.out_features)
        log_sigmas = log_sigmas * self.log_sigmas_in
        new_log_sigmas = insert_tensor(self.log_sigmas, pos, log_sigmas)
        self.out_features = new_centres.size(0)
        self.centres = nn.Parameter(new_centres)
        self.log_sigmas = nn.Parameter(new_log_sigmas)


    """        

    def forward(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
        return self.basis_func(distances)



# RBFs

def gaussian(alpha):
    phi = torch.exp(-1*alpha.pow(2))
    return phi

def linear(alpha):
    phi = alpha
    return phi

def quadratic(alpha):
    phi = alpha.pow(2)
    return phi

def inverse_quadratic(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
    return phi

def multiquadric(alpha):
    phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi

def inverse_multiquadric(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi

def spline(alpha):
    phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
    return phi

def poisson_one(alpha):
    phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
    return phi

def poisson_two(alpha):
    phi = ((alpha - 2*torch.ones_like(alpha)) / 2*torch.ones_like(alpha)) \
    * alpha * torch.exp(-alpha)
    return phi

def matern32(alpha):
    phi = (torch.ones_like(alpha) + 3**0.5*alpha)*torch.exp(-3**0.5*alpha)
    return phi

def matern52(alpha):
    phi = (torch.ones_like(alpha) + 5**0.5*alpha + (5/3) \
    * alpha.pow(2))*torch.exp(-5**0.5*alpha)
    return phi

def basis_func_dict():
    """
    A helper function that returns a dictionary containing each RBF
    """
    
    bases = {'gaussian': gaussian,
             'linear': linear,
             'quadratic': quadratic,
             'inverse quadratic': inverse_quadratic,
             'multiquadric': multiquadric,
             'inverse multiquadric': inverse_multiquadric,
             'spline': spline,
             'poisson one': poisson_one,
             'poisson two': poisson_two,
             'matern32': matern32,
             'matern52': matern52}
    return bases

def insert_tensor(tens, pos, elms):
    l = tens.size(0)
    a = torch.Tensor()
    b = tens
    c = []
    t = 0
    for _, p in enumerate(pos):
        if p <= l and p >= t:
            a, b = b.split((p - t, l - p), 0)
            t = p
            c.append(a)
            c.append(elms[_].unsqueeze(0))
        else:
            break
    c.append(b)
    return torch.cat(c, 0)
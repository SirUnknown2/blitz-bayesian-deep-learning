import unittest
import torch
from torch import nn
from torch.nn import functional as F

from blitz.modules.base_bayesian_module import BayesianModule
from blitz.modules.weight_sampler import TrainableRandomDistribution, PriorWeightDistribution

from typing import List

class BayesianConv1d(BayesianModule):

    # Implements Bayesian Conv2d layer, by drawing them using Weight Uncertanity on Neural Networks algorithm
    """
    Bayesian Linear layer, implements a Convolution 1D layer as proposed on Weight Uncertainity on Neural Networks
    (Bayes by Backprop paper).

    Its objective is be interactable with torch nn.Module API, being able even to be chained in nn.Sequential models with other non-this-lib layers

    parameters:
        in_channels: int -> incoming channels for the layer
        out_channels: int -> output channels for the layer
        kernel_size : tuple (int, int) -> size of the kernels for this convolution layer
        groups : int -> number of groups on which the convolutions will happend
        padding : int -> size of padding (0 if no padding)
        dilation int -> dilation of the weights applied on the input tensor


        bias: bool -> whether the bias will exist (True) or set to zero (False)
        prior_sigma_1: float -> prior sigma on the mixture prior distribution 1
        prior_sigma_2: float -> prior sigma on the mixture prior distribution 2
        prior_pi: float -> pi on the scaled mixture prior
        posterior_mu_init float -> posterior mean for the weight mu init
        posterior_rho_init float -> posterior mean for the weight rho init
        freeze: bool -> wheter the model will start with frozen(deterministic) weights, or not

    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 groups = 1,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 bias=True,
                 prior_sigma_1 = 0.1,
                 prior_sigma_2 = 0.002,
                 prior_pi = 1,
                 posterior_mu_init = 0,
                 posterior_rho_init = -7.0,
                 freeze = False,
                 prior_dist = None):
        super().__init__()

        #our main parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.freeze = freeze
        self.kernel_size = kernel_size
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias


        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        #parameters for the scale mixture prior
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.prior_dist = prior_dist

        #our weights
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size).normal_(posterior_mu_init, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size).normal_(posterior_rho_init, 0.1))
        self.weight_sampler = TrainableRandomDistribution(self.weight_mu, self.weight_rho)

        #our biases
        if self.bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_channels).normal_(posterior_mu_init, 0.1))
            self.bias_rho = nn.Parameter(torch.Tensor(out_channels).normal_(posterior_rho_init, 0.1))
            self.bias_sampler = TrainableRandomDistribution(self.bias_mu, self.bias_rho)
            self.bias_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        else:
            self.register_buffer('bias_zero', torch.zeros((self.out_channels)) )

        # Priors (as BBP paper)
        self.weight_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        self.bias_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x):
        #Forward with uncertain weights, fills bias with zeros if layer has no bias
        #Also calculates the complecity cost for this sampling
        if self.freeze:
            return self.forward_frozen(x)

        w = self.weight_sampler.sample()

        if self.bias:
            b = self.bias_sampler.sample()
            b_log_posterior = self.bias_sampler.log_posterior()
            b_log_prior = self.bias_prior_dist.log_prior(b)

        else:
            b = self.bias_zero
            b_log_posterior = 0
            b_log_prior = 0

        self.log_variational_posterior = self.weight_sampler.log_posterior() + b_log_posterior
        self.log_prior = self.weight_prior_dist.log_prior(w) + b_log_prior

        return F.conv1d(input=x,
                        weight=w,
                        bias=b,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups)

    def forward_frozen(self, x):
        # Computes the feedforward operation with the expected value for weight and biases (frozen-like)

        if self.bias:
            bias = self.bias_mu
            assert bias is self.bias_mu, "The bias inputed should be this layer parameter, not a clone."
        else:
            bias = self.bias_zero

        return F.conv1d(input=x,
                        weight=self.weight_mu,
                        bias=bias,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups)

class BayesianConv2d(BayesianModule):

    # Implements Bayesian Conv2d layer, by drawing them using Weight Uncertanity on Neural Networks algorithm
    """
    Bayesian Linear layer, implements a Convolution 2D layer as proposed on Weight Uncertainity on Neural Networks
    (Bayes by Backprop paper).

    Its objective is be interactable with torch nn.Module API, being able even to be chained in nn.Sequential models with other non-this-lib layers

    parameters:
        in_channels: int -> incoming channels for the layer
        out_channels: int -> output channels for the layer
        kernel_size : tuple (int, int) -> size of the kernels for this convolution layer
        groups : int -> number of groups on which the convolutions will happend
        padding : int -> size of padding (0 if no padding)
        dilation int -> dilation of the weights applied on the input tensor


        bias: bool -> whether the bias will exist (True) or set to zero (False)
        prior_sigma_1: float -> prior sigma on the mixture prior distribution 1
        prior_sigma_2: float -> prior sigma on the mixture prior distribution 2
        prior_pi: float -> pi on the scaled mixture prior
        posterior_mu_init float -> posterior mean for the weight mu init
        posterior_rho_init float -> posterior mean for the weight rho init
        freeze: bool -> wheter the model will start with frozen(deterministic) weights, or not

    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 groups = 1,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 bias=True,
                 prior_sigma_1 = 0.1,
                 prior_sigma_2 = 0.002,
                 prior_pi = 1,
                 posterior_mu_init = 0,
                 posterior_rho_init = -6.0,
                 freeze = False,
                 prior_dist = None):
        super().__init__()

        #our main parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.freeze = freeze
        self.kernel_size = (kernel_size, kernel_size)
        self.groups = groups
        self.stride = (stride, stride)
        self.padding = padding if isinstance(padding, str) else (padding, padding)
        self.dilation = (dilation, dilation)
        self.bias = bias


        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        #parameters for the scale mixture prior
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.prior_dist = prior_dist

        #our weights
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size).normal_(posterior_mu_init, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size).normal_(posterior_rho_init, 0.1))
        self.weight_sampler = TrainableRandomDistribution(self.weight_mu, self.weight_rho)

        #our biases
        if self.bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_channels).normal_(posterior_mu_init, 0.1))
            self.bias_rho = nn.Parameter(torch.Tensor(out_channels).normal_(posterior_rho_init, 0.1))
            self.bias_sampler = TrainableRandomDistribution(self.bias_mu, self.bias_rho)
            self.bias_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        else:
            self.register_buffer('bias_zero', torch.zeros((self.out_channels)) )

        # Priors (as BBP paper)
        self.weight_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x):
        #Forward with uncertain weights, fills bias with zeros if layer has no bias
        #Also calculates the complecity cost for this sampling
        if self.freeze:
            return self.forward_frozen(x)

        w = self.weight_sampler.sample()

        if self.bias:
            b = self.bias_sampler.sample()
            b_log_posterior = self.bias_sampler.log_posterior()
            b_log_prior = self.bias_prior_dist.log_prior(b)

        else:
            b = self.bias_zero
            b_log_posterior = 0
            b_log_prior = 0

        self.log_variational_posterior = self.weight_sampler.log_posterior() + b_log_posterior
        self.log_prior = self.weight_prior_dist.log_prior(w) + b_log_prior

        return F.conv2d(input=x,
                        weight=w,
                        bias=b,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups)

    def forward_frozen(self, x):
        # Computes the feedforward operation with the expected value for weight and biases (frozen-like)

        if self.bias:
            bias = self.bias_mu
            assert bias is self.bias_mu, "The bias inputed should be this layer parameter, not a clone."
        else:
            bias = self.bias_zero

        return F.conv2d(input=x,
                        weight=self.weight_mu,
                        bias=bias,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups)

class BayesianConvTranspose2d(BayesianModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 groups = 1,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 transposed = True,
                 output_padding = 0,
                 bias=True,
                 prior_sigma_1 = 0.1,
                 prior_sigma_2 = 0.002,
                 prior_pi = 1,
                 posterior_mu_init = 0,
                 posterior_rho_init = -6.0,
                 freeze = False,
                 prior_dist = None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.freeze = freeze
        self.kernel_size = (kernel_size, kernel_size)
        self.groups = groups
        self.stride = (stride, stride)
        self.padding = padding if isinstance(padding, str) else (padding, padding)
        self.dilation = (dilation, dilation)
        self.bias = bias
        self.transposed = transposed
        self.output_padding = output_padding


        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.prior_dist = prior_dist

        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size).normal_(posterior_mu_init, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size).normal_(posterior_rho_init, 0.1))
        self.weight_sampler = TrainableRandomDistribution(self.weight_mu, self.weight_rho)

        if self.bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_channels).normal_(posterior_mu_init, 0.1))
            self.bias_rho = nn.Parameter(torch.Tensor(out_channels).normal_(posterior_rho_init, 0.1))
            self.bias_sampler = TrainableRandomDistribution(self.bias_mu, self.bias_rho)
            self.bias_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        else:
            self.register_buffer('bias_zero', torch.zeros((self.out_channels)))

        self.weight_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def _output_padding(self, input, output_size, stride, padding, kernel_size, num_spatial_dims, dilation):

        if output_size is None:
            ret = (self.output_padding)
        else:
            has_batch_dim = input.dim() == num_spatial_dims + 2
            num_non_spatial_dims == 2 if has_batch_dim else 1
            if len(output_size) != num_non_spatial_dims + num_spatial_dims:
                output_size = output_size[num_non_spatial_dims:]
            if len(output_size) != num_spatial_dims:
                raise ValueError("BayesianConvTranspose2d: for {}d input, output_size must have {} or {} (got {})".format(input.dim(), num_spatial_dims, num_non_spatial_dims + num_spatial_dims, len(output_size)))
            min_sizes = torch.jit.annotate(List[int], [])
            max_sizes = torch.jit.annotate(List[int], [])

            for d in range(num_spatial_dims):
                dim_size = ((input.size(d + num_non_spatial_dims) - 1) * stride[d] - 2 * padding[d] + (dilation[d] if dilation is not None else 1) * (kernel_size[d] - 1) + 1)

                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]

                if size < min_size or size > max_size:
                    raise ValueError(("requested an output size of {}, but valid sizes range from {} to {} (for an input of {})").format(output_size, min_sizes, max_sizes, input.size()[2:]))

            res = torch.jit.annotate(List[int], [])
            for d in range(num_spatial_dims):
                res.append(output_size[d] - min_sizes[d])
            ret = res

        return ret

    def forward(self, x, output_size):

        assert isinstance(self.padding, tuple)
        num_spatial_dims = 2
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size, num_spatial_dims, self.dilation)

        if self.freeze:
            return self.forward_frozen(x, output_padding)

        w = self.weight_sampler.sample()

        if self.bias:
            b = self.bias_sampler.sample()
            b_log_posterior = self.bias_sampler.log_posterior()
            b_log_prior = self.bias_prior_dist.log_prior(b)
        else:
            b = self.bias_zero
            b_log_posterior = 0
            b_log_prior = 0

        self.log_variational_posterior = self.weight_sampler.log_posterior() + b_log_posterior
        self.log_prior = self.weight_prior_dist.log_prior(w) + b_log_prior

        return F.conv_transpose2d(x, w, b, self.stride, self.padding, output_padding, self.groups, self.dilation)

    def forward_frozen(self, x, output_padding):

        if self.bias:
            bias = self.bias_mu
            assert bias is self.bias_mu, "The bias inputed should be this layer parameter, not a clone."

        else:
            bias = self.bias_zero

        return F.conv_transpose2d(x, w, b, self.stride, self.padding, output_padding, self.groups, self.dilation)


class BayesianConv3d(BayesianModule):

    # Implements Bayesian Conv2d layer, by drawing them using Weight Uncertanity on Neural Networks algorithm
    """
    Bayesian Linear layer, implements a Convolution 3D layer as proposed on Weight Uncertainity on Neural Networks
    (Bayes by Backprop paper).

    Its objective is be interactable with torch nn.Module API, being able even to be chained in nn.Sequential models with other non-this-lib layers

    parameters:
        in_channels: int -> incoming channels for the layer
        out_channels: int -> output channels for the layer
        kernel_size : tuple (int, int) -> size of the kernels for this convolution layer
        groups : int -> number of groups on which the convolutions will happend
        padding : int -> size of padding (0 if no padding)
        dilation int -> dilation of the weights applied on the input tensor


        bias: bool -> whether the bias will exist (True) or set to zero (False)
        prior_sigma_1: float -> prior sigma on the mixture prior distribution 1
        prior_sigma_2: float -> prior sigma on the mixture prior distribution 2
        prior_pi: float -> pi on the scaled mixture prior
        posterior_mu_init float -> posterior mean for the weight mu init
        posterior_rho_init float -> posterior mean for the weight rho init
        freeze: bool -> wheter the model will start with frozen(deterministic) weights, or not

    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 groups = 1,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 bias=True,
                 prior_sigma_1 = 0.1,
                 prior_sigma_2 = 0.002,
                 prior_pi = 1,
                 posterior_mu_init = 0,
                 posterior_rho_init = -6.0,
                 freeze = False,
                 prior_dist = None):
        super().__init__()

        #our main parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.freeze = freeze
        self.kernel_size = kernel_size
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias


        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        #parameters for the scale mixture prior
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.prior_dist = prior_dist

        #our weights
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size).normal_(posterior_mu_init, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size).normal_(posterior_rho_init, 0.1))
        self.weight_sampler = TrainableRandomDistribution(self.weight_mu, self.weight_rho)

        #our biases
        if self.bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_channels).normal_(posterior_mu_init, 0.1))
            self.bias_rho = nn.Parameter(torch.Tensor(out_channels).normal_(posterior_rho_init, 0.1))
            self.bias_sampler = TrainableRandomDistribution(self.bias_mu, self.bias_rho)
            self.bias_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        else:
            self.register_buffer('bias_zero', torch.zeros((self.out_channels)) )

        # Priors (as BBP paper)
        self.weight_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x):
        #Forward with uncertain weights, fills bias with zeros if layer has no bias
        #Also calculates the complecity cost for this sampling
        if self.freeze:
            return self.forward_frozen(x)

        w = self.weight_sampler.sample()

        if self.bias:
            b = self.bias_sampler.sample()
            b_log_posterior = self.bias_sampler.log_posterior()
            b_log_prior = self.bias_prior_dist.log_prior(b)

        else:
            b = self.bias_zero
            b_log_posterior = 0
            b_log_prior = 0

        self.log_variational_posterior = self.weight_sampler.log_posterior() + b_log_posterior
        self.log_prior = self.weight_prior_dist.log_prior(w) + b_log_prior

        return F.conv3d(input=x,
                        weight=w,
                        bias=b,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups)

    def forward_frozen(self, x):
        # Computes the feedforward operation with the expected value for weight and biases (frozen-like)

        if self.bias:
            bias = self.bias_mu
            assert bias is self.bias_mu, "The bias inputed should be this layer parameter, not a clone."
        else:
            bias = self.bias_zero

        return F.conv3d(input=x,
                        weight=self.weight_mu,
                        bias=bias,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups)

import torch
import torch.nn as nn

import math

import numpy as np

import torch.nn.functional


def modulate(x, shift, scale):
    return x * (1 + scale) + shift



#########################MODEL FOR DENOISER###########################
class EDMPrecond(torch.nn.Module):
    def __init__(self,
        channels = 3, 
        use_fp16 = False,
        sigma_min = 0,
        sigma_max = float('inf'),
        sigma_data  = 1,
        depth = 6,
        network = None,
    ):
        super().__init__()

        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.sigma_data = sigma_data
        if network is not None:
            self.model = network
        else:
            self.model = MLP(channels=channels, hidden_size=512, depth=depth)

    def forward(self, x, sigma, force_fp32=False, **model_kwargs):

        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
    
        F_x = self.model((c_in * x).to(dtype), c_noise, **model_kwargs).to(dtype)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)

        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    #@torch.no_grad()
    def sample(self, cond=None, batch_seeds=None, channels=3, num_steps=18):

        device = batch_seeds.device
        batch_size = batch_seeds.shape[0]

        rnd = None
        points = batch_seeds

        latents = points.float().to(device)

        points = edm_sampler(self, latents, cond, num_steps=num_steps)
        return points

    @torch.no_grad()
    def inverse(self, cond=None, samples=None, channels=3, num_steps=18):
        return inverse_edm_sampler(self, samples, cond, num_steps=num_steps)


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):  
    # disable S_churn
    assert S_churn==0

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    outputs = []
    outputs.append((x_next / t_steps[0]).detach().cpu().numpy())
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        # x_hat = x_cur
        t_hat = t_cur

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        outputs.append((x_next / (1+t_next**2).sqrt()).detach().cpu().numpy())
    return x_next, outputs

def inverse_edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):  
    # disable S_churn
    assert S_churn==0

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])+1e-8]) # t_N = 0
    t_steps = torch.flip(t_steps, [0])#[1:]

    # Main sampling loop.
    x_next = latents.to(torch.float64)# * t_steps[0]

    # outputs = []
    outputs = None
    # outputs.append((x_next / t_steps[0]).detach().cpu().numpy())

    #print(t_steps[0])
    #print(x_next.mean(), x_next.std())
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        # print('steps', t_cur, t_next)
        x_cur = x_next
        # print('cur', (x_cur / t_cur).mean(), (x_cur / t_cur).std())

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        x_hat = x_cur
        t_hat = t_cur

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        #print('next', (x_next / (1+t_next**2).sqrt()).mean(), (x_next / (1+t_next**2).sqrt()).std())

        # outputs.append((x_next / (1+t_next**2).sqrt()).detach().cpu().numpy())
    x_next = x_next / (1+t_next**2).sqrt()
    return x_next, outputs



#############################NETWORK ARCHITECHTUURE##############################



class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)
    


def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596

def mp_sum(a, b, t=0.5):
    # print(a.mean(), a.std(), b.mean(), b.std())
    return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t ** 2)

class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w)) # forced weight normalization
        w = normalize(w) # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel())) # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1]//2,))

class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128, other_dim=0):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        # self.mlp = nn.Linear(self.embedding_dim+3, dim)/
        self.mlp = MPConv(self.embedding_dim+3+other_dim, dim, kernel=[])

    @staticmethod
    def embed(input, basis):
        # print(input.shape, basis.shape)
        projections = torch.einsum('nd,de->ne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=1)
        return embeddings
    
    def forward(self, input):
        # input: N x 3
        if input.shape[1] != 3:
            input, others = input[:, :3], input[:, 3:]
        else:
            others = None
        
        if others is None:
            embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=1)) # N x C
        else:
            embed = self.mlp(torch.cat([self.embed(input, self.basis), input, others], dim=1))
        return embed


class Network(nn.Module):
    def __init__(
        self,
        channels = 3,
        hidden_size = 512,
        depth = 6,
    ):
        super().__init__()

        self.emb_fourier = MPFourier(hidden_size)
        self.emb_noise = MPConv(hidden_size, hidden_size, kernel=[])

        self.x_embedder = PointEmbed(dim=hidden_size, other_dim=channels-3)

        self.gains = nn.ParameterList([
            torch.nn.Parameter(torch.zeros([])) for _ in range(depth)
        ])
        ##
        self.layers = nn.ModuleList([
            nn.ModuleList([
                MPConv(hidden_size, hidden_size, []),
                MPConv(hidden_size, hidden_size, []),
                MPConv(hidden_size, 1 * hidden_size, []),
            ]) for _ in range(depth)
        ])


        self.final_emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.final_out_gain = torch.nn.Parameter(torch.zeros([]))
        self.final_layer = nn.ModuleList([
            MPConv(hidden_size, hidden_size, []),
            MPConv(hidden_size, channels, []),
            MPConv(hidden_size, hidden_size, []),
        ])

        self.res_balance = 0.3


    def forward(self, x, t):
        x = self.x_embedder(x)
        if t.ndim == 1:
            t = t.repeat(x.shape[0])

        t = mp_silu(self.emb_noise(self.emb_fourier(t.flatten())))

        for (x_proj_pre, x_proj_post, emb_linear), emb_gain in zip(self.layers, self.gains):

            c = emb_linear(t, gain=emb_gain) + 1

            x = normalize(x)
            y = x_proj_pre(mp_silu(x))
            y = mp_silu(y * c.to(y.dtype))
            y = x_proj_post(y)
            x = mp_sum(x, y, t=self.res_balance)

        x_proj_pre, x_proj_post, emb_linear = self.final_layer
        c = emb_linear(t, gain=self.final_emb_gain) + 1
        y = x_proj_pre(mp_silu(normalize(x)))
        y = mp_silu(y * c.to(y.dtype))
        out = x_proj_post(y, gain=self.final_out_gain)
    
        return out
    
    
    
    



class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x): 
        return torch.sigmoid(x) * x


class RandomFourierFeatures(nn.Module):
    def __init__(self, input_dim, output_dim, scale=1.0):
        super(RandomFourierFeatures, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scale = scale
        self.B = nn.Parameter(self.scale * torch.randn(self.input_dim, self.output_dim // 2), requires_grad=False)
        

    def forward(self, x):
        x_proj = x @ self.B
        x_proj = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return x_proj


class FourierFeatsEncoding(nn.Module):
    """Multi-scale sinusoidal encodings. Support ``integrated positional encodings`` if covariances are provided.
    Each axis is encoded with frequencies ranging from 2^min_freq_exp to 2^max_freq_exp.

    Args:
        in_dim: Input dimension of tensor
        num_frequencies: Number of encoded frequencies per axis
        include_input: Append the input coordinate to the encoding
    """

    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        include_input: bool = False
    ) -> None:
        super(FourierFeatsEncoding, self).__init__()

        assert in_dim > 0, "in_dim should be greater than zero"
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.min_freq = 0.0
        self.max_freq = num_frequencies - 1.0
        self.include_input = include_input

    def get_out_dim(self) -> int:
        assert self.in_dim is not None, "Input dimension has not been set"
        out_dim = self.in_dim * self.num_frequencies * 2
        if self.include_input:
            out_dim += self.in_dim
        return out_dim

    def forward(
        self,
        in_tensor
    ):
        """Calculates NeRF encoding. 

        Args:
            in_tensor: For best performance, the input tensor should be between 0 and 1.
        Returns:
            Output values will be between -1 and 1
        """
        scaled_in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies).to(in_tensor.device)
        scaled_inputs = scaled_in_tensor[..., None] * freqs  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)  # [..., "input_dim" * "num_scales"]

        encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1))

        if self.include_input:
            encoded_inputs = torch.cat([in_tensor, encoded_inputs], dim=-1)
        return encoded_inputs



class MLP(nn.Module):
    def __init__(self,
        channels = 3,
        hidden_size = 256,
        depth = 6,):
    #input_dim: int = 3, time_dim: int = 1, hidden_dim: int = 128, fourier_encoding: str = 'FF', fourier_dim: int = 0):
        super().__init__()
        self.input_dim = channels
        self.hidden_dim = hidden_size
        self.ff_module = FourierFeatsEncoding(in_dim=4, num_frequencies=6, include_input=True)
        self.fourier_dim = ((3+1) * 6 * 2) + (3 + 1)
        self.rff_module = nn.Identity()

        self.main = nn.Sequential(
            nn.Linear(self.fourier_dim, self.hidden_dim),
            Swish(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            Swish(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            Swish(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            Swish(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )
    

    def forward(self, x, t):
        sz = x.size()
        t = t.reshape(-1, 1)        
        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        
        h = torch.cat([x, t], dim=1)
        
        h = self.rff_module(h)
        h = self.ff_module(h)
       
        output = self.main(h)
        output = output.reshape(*sz)
        
        return output


class MLP_dd(nn.Module):
    def __init__(self,
        channels = 3,
        hidden_size = 256,
        depth = 6,):
    #input_dim: int = 3, time_dim: int = 1, hidden_dim: int = 128, fourier_encoding: str = 'FF', fourier_dim: int = 0):
        super().__init__()
        self.input_dim = channels
        self.hidden_dim = hidden_size
        self.ff_module = FourierFeatsEncoding(in_dim=4, num_frequencies=6, include_input=True)
        self.fourier_dim = ((3+1) * 6 * 2) + (3 + 1) + 5
        self.rff_module = nn.Identity()

        self.main = nn.Sequential(
            nn.Linear(self.fourier_dim, self.hidden_dim),
            Swish(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            Swish(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            Swish(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            Swish(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )
    

    def forward(self, x, t):
        sz = x.size()
        t = t.reshape(-1, 1)        
        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        xyz=x[:,:3]
        h = torch.cat([xyz, t], dim=1)
        
        h = self.rff_module(h)
        h = self.ff_module(h)
       

        output = self.main(torch.cat([h, x[:,3:]], dim=1))
        #output = output.reshape((-1,8))
        
        return output




class FMCond(torch.nn.Module):
    def __init__(self,
        channels = 3, 
        use_fp16 = False,
        sigma_min = 0,
        sigma_max = float('inf'),
        sigma_data  = 1,
        depth = 6,
        network = None,
    ):
        super().__init__()

        self.use_fp16 = use_fp16

        if network is not None:
            self.net = network
        else:
            self.net = MLP(channels=channels, hidden_size=512, depth=depth)

    def forward(self, x, sigma):
        x = x
        sigma = sigma.to(torch.float32).reshape(-1, 1)
        V_x = self.net(x, sigma).to(torch.float32)

        return V_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    #@torch.no_grad()
    def sample(self, cond=None, batch_seeds=None, channels=3, num_steps=64,enable_grad=False, intermediate=False):

        device = batch_seeds.device
        batch_size = batch_seeds.shape[0]

        rnd = None
        points = batch_seeds

        latents = points.float().to(device)

        if intermediate:
            sample,sol=ot_sampler(self.net, latents, num_steps=num_steps,enable_grad=enable_grad, intermediate=intermediate)
            return sample, sol
        else:
            sample=ot_sampler(self.net, latents, num_steps=num_steps,enable_grad=enable_grad,intermediate=intermediate)
            return sample
    
    def inverse(self, cond=None, samples=None, channels=3, num_steps=18,enable_grad=False,intermediate=False):

        device = samples.device
        batch_size = samples.shape[0]

        rnd = None
        points = samples

        latents = points.float().to(device)

        if intermediate:
            sample,sol=ot_inverse(self.net, latents, num_steps=num_steps,enable_grad=enable_grad, intermediate=intermediate)
            
            return sample, sol
        else:
            sample=ot_inverse(self.net, latents, num_steps=num_steps,enable_grad=enable_grad, intermediate=intermediate)
            
            return sample
        
        
from flow_matching.solver import ODESolver

def ot_sampler(
    net, latents,num_steps=18, enable_grad=False, intermediate=False):  

    # Time step discretization.
    t_steps = torch.linspace(0, 1, num_steps+1)
    # Main sampling loop.
    solver = ODESolver(velocity_model=net)
    solutions = solver.sample(time_grid=t_steps, x_init=latents, method='midpoint', step_size=1/num_steps, return_intermediates=intermediate,enable_grad=enable_grad)

    if intermediate:
        return solutions[-1], solutions
    else:
        return solutions

def ot_inverse(    net, sample,
    num_steps=18, enable_grad=False, intermediate=False):
    
    # Time step discretization.
    t_steps = torch.linspace(0, 1, num_steps+1)
    # Main sampling loop.
    inverse_net=InverseModel(net)
    solver = ODESolver(velocity_model=inverse_net)
    solutions = solver.sample(time_grid=t_steps, x_init=sample, method='midpoint', step_size=1/num_steps, return_intermediates=intermediate,enable_grad=enable_grad)

    if intermediate:
        
        return solutions[-1], solutions
    else:
        return solutions



class InverseModel(torch.nn.Module):
    def __init__(self, vector_field):
        super(InverseModel, self).__init__()
        self.vector_field = vector_field

    def forward(self, x,t):
        if torch.allclose(t, torch.ones_like(t)):
            return torch.zeros_like(x)
        return -self.vector_field(x,1-t)
    



###########àà

import tinycudann as tcnn



class MLP_tiny(nn.Module):
    def __init__(self,
        channels = 3,
        hidden_size = 256,
        depth = 6,):
    #input_dim: int = 3, time_dim: int = 1, hidden_dim: int = 128, fourier_encoding: str = 'FF', fourier_dim: int = 0):
        super().__init__()
        self.input_dim = channels
        self.hidden_dim = hidden_size
        
        self.config={
            "encoding": {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 16,
                "base_resolution": 16,
                "per_level_scale": 2
            },
            "network": {
                "otype": "FullyFusedMLP",
                "activation": "ReLu",
                "output_activation": "None",
                "n_neurons": 128,
                "n_hidden_layers": 6
            }
        }
        self.fourier_dim = ((3+1) * 6 * 2) + (3 + 1)
        self.ff_module = FourierFeatsEncoding(in_dim=4, num_frequencies=6, include_input=True)
        self.rff_module = nn.Identity()


        #self.main = tcnn.NetworkWithInputEncoding(4, 3,self.config["encoding"], self.config["network"])
        self.encoding = tcnn.Encoding(4, self.config["encoding"])
        #self.main =  tcnn.Network(self.encoding.n_output_dims+4, 3, self.config["network"])

    
        self.main = nn.Sequential(
                nn.Linear(self.encoding.n_output_dims+ self.fourier_dim, self.hidden_dim),
                Swish(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                Swish(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                Swish(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                Swish(),
                nn.Linear(self.hidden_dim, self.input_dim),
            )
    def forward(self, x, t):
        sz = x.size()
        t = t.reshape(-1, 1)        
        t = t.reshape(-1, 1).expand(x.shape[0], 1)

        h = torch.cat([x, t], dim=1)
        
        h_ff = self.ff_module(h)
        h_ngp=self.encoding(h).to(torch.float32)
        
        output = self.main(torch.cat([h_ff, h_ngp], dim=1))
        output = output.reshape(*sz)
        
        return output
    
    
########################### DEFORMATION PYRAMYD ##########################


class FMCond_NDP(torch.nn.Module):
    def __init__(self,
        channels = 3, 
        use_fp16 = False,
        sigma_min = 0,
        sigma_max = float('inf'),
        sigma_data  = 1,
        depth = 6,
        network = None,
    ):
        super().__init__()

        self.use_fp16 = use_fp16

        if network is not None:
            self.net = network
        else:
            self.net = MLP(channels=channels, hidden_size=512, depth=depth)

    def forward(self, x, sigma,max_level=0, min_level=0):
        x = x
        sigma = sigma.to(torch.float32).reshape(-1, 1)
        V_x = self.net(x=x, t=sigma, max_level=max_level, min_level=min_level).to(torch.float32)

        return V_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    #@torch.no_grad()
    def sample(self, cond=None, batch_seeds=None, channels=3, num_steps=64):

        device = batch_seeds.device
        batch_size = batch_seeds.shape[0]

        rnd = None
        points = batch_seeds

        latents = points.float().to(device)

        sample,sol=ot_sampler(self.net, latents, cond, num_steps=num_steps)
        
        return sample.to(torch.double), sol
    @torch.no_grad()
    
    def inverse(self, cond=None, samples=None, channels=3, num_steps=18):

        device = samples.device
        batch_size = samples.shape[0]

        rnd = None
        points = samples

        latents = points.float().to(device)

        sample,sol=ot_inverse(self.net, latents, cond, num_steps=num_steps)
        
        return sample.to(torch.double), sol


class Deformation_Pyramid(nn.Module):
    def __init__(self, depth=4, width=256, device='cuda:0', k0=-9, m=9):
        super(Deformation_Pyramid, self).__init__()
        pyramid = []

        for i in range (m):
            pyramid.append(
                NDPLayer(depth,
                         width,
                         k0,
                         i,
                         ).to(device)
            )
        self.pyramid = pyramid
        self.n_hierarchy = m
        self.current_max=0
    def forward(self, x,t, max_level=None, min_level=0):
        if max_level is None:
            max_level = self.current_max
        assert max_level < self.n_hierarchy, "more level than defined"
        sz = x.size()
        t = t.reshape(-1, 1)        
        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        x_in = x
        input=torch.cat([x_in, t], dim=-1)
        out_ini= 0
        for i in range(min_level, max_level + 1):
            out_ini =  out_ini+0.01*self.pyramid[i](input)
        return out_ini

    def gradient_setup(self, optimized_level):

        assert optimized_level < self.n_hierarchy, "more level than defined"

        # optimize current level, freeze the other levels
        for i in range( self.n_hierarchy):
            net = self.pyramid[i]
            if i == optimized_level:
                for param in net.parameters():
                    param.requires_grad = True
            else:
                for param in net.parameters():
                    param.requires_grad = False



class NDPLayer(nn.Module):
    def __init__(self, depth, width, k0, m):
        super().__init__()

        self.k0 = k0
        self.m = m
        self.ff_module = FourierFeatsEncoding(in_dim=4, num_frequencies=m, include_input=True)
        self.fourier_dim = ((3+1) * m * 2) + (3 + 1)

        self.input= nn.Sequential( nn.Linear( self.fourier_dim,width), Swish())
        self.mlp = MLP_ndp(depth=depth,width=width)

        self.out_branch = nn.Linear(width, 3) # scale branch

        self._reset_parameters()

    def forward (self, x):

        fea = self.ff_module( x )
        fea = self.input(fea)
        fea = self.mlp(fea)

        return self.out_branch(fea)
    

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

import torch.nn.functional as F   
class MLP_ndp(torch.nn.Module):
    def __init__(self, depth, width):
        super().__init__()
        self.pts_linears = nn.ModuleList( [nn.Linear(width, width) for i in range(depth - 1)])
        self.act = Swish()
    def forward(self, x):
        for i, l in enumerate(self.pts_linears):
            x = self.pts_linears[i](x)
            x = self.act(x)
        return x
import torch
from . import coefficients as cf

T0 = 0.154
T0p = 0.200


def _as_tensor(value, device, dtype=torch.float64):
    return torch.as_tensor(value, dtype=dtype, device=device)

CHI_KEYS = {
    (0,0,0): "chi0", 
    (2,0,0): "chi2B",
    (0,2,0): "chi2Q",
    (0,0,2): "chi2S",
    (1,1,0): "chi11BQ",
    (1,0,1): "chi11BS",
    (0,1,1): "chi11QS",
    (4,0,0): "chi4B",
    (0,4,0): "chi4Q",
    (0,0,4): "chi4S",
    (3,1,0): "chi31BQ",
    (3,0,1): "chi31BS",
    (1,3,0): "chi13BQ",
    (1,0,3): "chi13BS",
    (2,2,0): "chi22BQ",
    (2,0,2): "chi22BS",
    (2,1,1): "chi211BQS",
    (1,2,1): "chi121BQS",
    (1,1,2): "chi112BQS"
}

class ChiRatioParam:
    """General chi_{ijk} parametrization"""
    def __init__(self, name, device='cpu'):
        self.device = device
        self.A = _as_tensor(cf.COEFFICIENTS_A[name], device=device)
        self.B = _as_tensor(cf.COEFFICIENTS_B[name], device=device)
        self.c0 = _as_tensor(cf.COEFFICIENTS_C0.get(name, 0.0), device=device)

    def __call__(self, T):
        T = _as_tensor(T, device=self.device)
        T = torch.atleast_1d(T)
        t = T / T0
        inv = 1.0 / t
        powers = torch.arange(len(self.A), dtype=T.dtype, device=self.device)

        inv_powers = inv[:, None] ** powers[None, :]
        num = torch.sum(self.A[None, :] * inv_powers, dim=1)
        denom = torch.sum(self.B[None, :] * inv_powers, dim=1)
        return num / denom + self.c0

class Chi2BParam:
    """Special chi_2^B parametrization"""
    def __init__(self, device='cpu'):
        self.device = device
        params = cf.CHI2B_SPECIAL
        self.h1 = _as_tensor(params["h1"], device=device)
        self.h2 = _as_tensor(params["h2"], device=device)
        self.f3 = _as_tensor(params["f3"], device=device)
        self.f4 = _as_tensor(params["f4"], device=device)
        self.f5 = _as_tensor(params["f5"], device=device)

    def __call__(self, T):
        T = _as_tensor(T, device=self.device)
        T = torch.atleast_1d(T)
        tp = T / T0p
        return torch.exp(-self.h1 / tp - self.h2 / (tp ** 2)) * self.f3 * (1.0 + torch.tanh(self.f4 * tp + self.f5))

def chi(i, j, k, T, device='cpu'):
    """Return chi_{ijk}(T) as a torch tensor on the specified device"""
    if (i,j,k) == (2,0,0):
        return Chi2BParam(device=device)(T)

    key = CHI_KEYS.get((i,j,k))
    if key is None:
        return torch.zeros_like(T, dtype=torch.float64, device=device)

    return ChiRatioParam(key, device=device)(T)

def chi_000(T, device='cpu'):
    return chi(0, 0, 0, T, device=device)

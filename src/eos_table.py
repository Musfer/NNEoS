import os

import torch
import torch.nn as nn

from .chi_module import ChiRatioParam

LQCD_CONVERSION = 5.068 ** 3
_BETA = 3.845630982896477
_DELTA_T = 0.05


def _resolve_device(value):
    if isinstance(value, torch.Tensor):
        return value.device
    return torch.device("cpu")


def Plat(T, muB=0.0, muQ=0.0, muS=0.0):
    """Lattice QCD pressure P(T) at mu=0 in units of [GeV fm^{-3}]."""
    device = _resolve_device(T)
    T = torch.as_tensor(T, dtype=torch.float64, device=device)
    chi_param = ChiRatioParam("chi0", device=device)
    p = chi_param(T)
    return p * T**4 * LQCD_CONVERSION


def entropy(T, muB=0.0, muQ=0.0, muS=0.0, dT=1e-4):
    """Entropy density s = dP/dT via central finite difference [fm^{-3}]."""
    device = _resolve_device(T)
    T = torch.as_tensor(T, dtype=torch.float64, device=device)
    return (Plat(T + dT) - Plat(T - dT)) / (2 * dT)


class ResidualBlock(nn.Module):
    """Single residual layer: x -> x + tanh(Wx + b)."""

    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.act = nn.Tanh()

    def forward(self, x):
        return x + self.act(self.fc(x))


class EOSNetDeep(nn.Module):
    """
    Deep residual network for the equation of state.

    Input:  (T, muB, muQ, muS)  in GeV
    Output: scalar residual pressure  p_res / [beta*(T+deltaT)^2]
    """

    def __init__(self, input_dim=4, hidden_dim=64, output_dim=1, n_layers=16):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(n_layers)])
        self.out_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.tanh(self.input_layer(x))
        for block in self.blocks:
            x = block(x)
        return self.out_layer(x)


def load_eos_model(filename, device="cpu", hidden_dim=128, n_layers=24):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No saved model found at: {filename}")

    model = EOSNetDeep(input_dim=4, hidden_dim=hidden_dim, output_dim=1, n_layers=n_layers).to(device)
    model.load_state_dict(torch.load(filename, map_location=device))
    model.eval()
    return model


def generate_obs(X_tensor, modelForward, device="cpu"):
    """
    Return (eps, nB, nQ, nS, P) for a batch of thermodynamic states.
    """
    if isinstance(X_tensor, torch.Tensor):
        X_tensor = X_tensor.detach().to(device=device).requires_grad_(True)
    else:
        X_tensor = torch.tensor(X_tensor, dtype=torch.float32, device=device, requires_grad=True)

    with torch.enable_grad():
        y_pred = modelForward(X_tensor)
        dP_dX = torch.autograd.grad(
            outputs=y_pred[:, 0].sum(),
            inputs=X_tensor,
            create_graph=False,
        )[0]

    T = X_tensor[:, 0].detach()
    muB = X_tensor[:, 1].detach()
    muQ = X_tensor[:, 2].detach()
    muS = X_tensor[:, 3].detach()

    scale = _BETA * (T + _DELTA_T) ** 2
    dtype = y_pred.dtype

    nB = dP_dX[:, 1].detach() * scale
    nQ = dP_dX[:, 2].detach() * scale
    nS = dP_dX[:, 3].detach() * scale

    P = y_pred[:, 0].detach() * scale + Plat(T).to(device=device, dtype=dtype)
    s = (
        dP_dX[:, 0].detach() * scale
        + 2 * y_pred[:, 0].detach() * _BETA * (T + _DELTA_T)
        + entropy(T).to(device=device, dtype=dtype)
    )

    eps = T * s + muB * nB + muQ * nQ + muS * nS - P

    return torch.stack((eps, nB, nQ, nS, P), dim=1)
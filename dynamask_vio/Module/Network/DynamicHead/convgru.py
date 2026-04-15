import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    def __init__(self, hidden_dim: int = 128, input_dim: int = 128, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        gate_dim = hidden_dim + input_dim

        self.convz = nn.Conv2d(gate_dim, hidden_dim, kernel_size, padding=padding)
        self.convr = nn.Conv2d(gate_dim, hidden_dim, kernel_size, padding=padding)
        self.convq = nn.Conv2d(gate_dim, hidden_dim, kernel_size, padding=padding)

    def forward(self, h_prev: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        hx = torch.cat([h_prev, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h_prev, x], dim=1)))
        return (1.0 - z) * h_prev + z * q


class ConvLSTMCell(nn.Module):
    def __init__(self, hidden_dim: int = 128, input_dim: int = 128, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        gate_dim = hidden_dim + input_dim
        self.conv = nn.Conv2d(gate_dim, 4 * hidden_dim, kernel_size, padding=padding)
        self.hidden_dim = hidden_dim

    def forward(self, h_prev: torch.Tensor, c_prev: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gates = self.conv(torch.cat([h_prev, x], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c_prev + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

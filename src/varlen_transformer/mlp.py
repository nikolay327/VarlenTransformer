import torch.nn.functional as F
from torch import Tensor, nn

from ._validation import positive_int


class MLP(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int):
        super(MLP, self).__init__()

        positive_int("emb_dim", emb_dim)
        positive_int("hidden_dim", hidden_dim)

        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, emb_dim)

    def forward(self, x: Tensor):
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.fc2(x)
        return x

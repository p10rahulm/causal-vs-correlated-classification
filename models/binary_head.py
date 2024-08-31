import torch.nn as nn
import torch.nn.functional as F


class MultiHeadBinaryModel(nn.Module):
    def __init__(self, input_dim=393216, hidden_dim_1=1000, hidden_dim_2=10, l=37):
        super(MultiHeadBinaryModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.out_dim = 1
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim_1)
        self.fc2 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
        self.l = l

        for i in range(1, self.l + 1):
            setattr(self, f"out{i}", nn.Linear(self.hidden_dim_2, self.out_dim))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        out_list = []
        for i in range(1, self.l + 1):
            var = getattr(self, f"out{i}")
            out_list.append(F.sigmoid(var(x)))

        return [out for out in out_list]

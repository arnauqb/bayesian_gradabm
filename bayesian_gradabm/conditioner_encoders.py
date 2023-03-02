import torch


class RNN(torch.nn.Module):
    """
    RNN that encondes conditioning data.
    """

    def __init__(
        self, input_size=1, hidden_size=32, num_layers=2, output_size=16, n_timesteps=50
    ):
        super().__init__()
        self.torch_rnn = torch.nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            bidirectional=False,
        )
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.torch_rnn(x)
        return self.fc(x[:, -1, :])

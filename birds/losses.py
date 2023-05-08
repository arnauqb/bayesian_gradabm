import torch
import warnings

def create_mmd_loss(y):

    """
    Assumes y is a torch.tensor of shape (T, C) containing the C-dimensional time series of length T
    """

    if len(y.shape) == 1:
        y = y.unsqueeze(-1)
    y_matrix = y.unsqueeze(0)
    y_sigma = torch.median(torch.pow(torch.cdist(y_matrix, y_matrix), 2))
    T = y.shape[0]
    ny = T
    kyy = (torch.exp( - torch.pow(torch.cdist(y_matrix, y_matrix), 2) / y_sigma ) - torch.eye(ny)).sum() / (ny * (ny - 1))

    def mmd_loss(_, x):

        """
        Assumes x, y are shape (T, C)
        """
        nx = x.shape[0]
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)
        x_matrix = x.unsqueeze(0)
        kxx = torch.exp( - torch.pow(torch.cdist(x_matrix, x_matrix), 2) / y_sigma )
        kxx = (kxx - torch.eye(nx)).sum() / (nx * (nx - 1))
        kxy = torch.exp( - torch.pow(torch.cdist(x_matrix, y_matrix), 2) / y_sigma )
        kxy = kxy.mean()
        kxynan = torch.isnan(kxy)
        if kxynan:
            warnings.warn("kxy nan")
        kxxnan = torch.isnan(kxx)
        if kxxnan:
            warnings.warn("kxx nan")
        return kxx + kyy - 2 * kxy

    return mmd_loss

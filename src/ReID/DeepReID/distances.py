import torch

def L2_distance(x:torch.Tensor, y: torch.Tensor, squared_distance:bool=False):
    """
    Args:
        x: must be shaped [k1,d] 
        y: must be shaped [k2,d]
        squared_distance: if set to true, final square root is not computed.

    Returns:
        torch.Tensor shaped [k1,k2]
    """
    assert x.dim() == 2 and y.dim() == 2
    assert x.shape[1] == y.shape[1]

    x = x.reshape(x.shape[0], 1, x.shape[1])
    y = y.reshape(1, y.shape[0], y.shape[1])
    if squared_distance:
        return torch.sum(torch.pow(x-y,2), -1)
    else:
        return torch.sqrt(torch.sum(torch.pow(x-y,2), -1).clamp(min=1e-12))
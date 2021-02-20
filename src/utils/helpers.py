import torch
import torchvision


class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


def binarize_masks(masks):
    bmasks = torch.tensor(masks)
    bmasks[masks > 0] = 1
    bmasks[masks <= 0] = 0

    return bmasks


def create_3channel_masks(masks):
    assert(len(masks.size()[1:]) == 2)
    return masks.unsqueeze(1).repeat(1, 3, 1, 1)

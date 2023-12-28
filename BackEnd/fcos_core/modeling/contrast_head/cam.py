import torch
from pytorch_grad_cam import GradCAM


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()

class CAMModule(torch.nn.Module):
    """
    Module for DDTNet computation. Takes feature maps from the backbone and
    DDTNet outputs and losses.
    """

    def __init__(self, cfg):
        super(CAMModule, self).__init__()
        self.target_layer = cfg.MODEL.CAM.TARGET_LAYER
        
import numpy as np
import torch
import torch.nn as nn

from torchvision.ops.boxes import nms
from torchvision.ops.roi_pool import RoIPool
from torchvision.ops.roi_align import RoIAlign

def bilinear_interpolate(img, x, y):

    """
    * 双线性插值
    """

    x0 = np.floor(x)
    y0 = np.floor(y)

    x1 = np.ceil(x)
    y1 = np.ceil(y)

    x0 = np.clip(x0, 0, img.size(1) - 1) # h w
    y0 = np.clip(y0, 0, img.size(0) - 1)

    x1 = np.clip(x1, 0, img.size(1) - 1)
    y1 = np.clip(y1, 0, img.size(0) - 1)

    pa = img[y0, x0]  # pixel left-top
    pb = img[y1, x0]  # pixel right-top
    pc = img[y0, x1]
    pd = img[y1, x1]




class TorchROIPool(object):

    def __init__(self, output_size, scaling_factor):

        self.output_size = output_size
        self.scaling_factor = scaling_factor

    def _roi_pool(self, features):

        num_channels, h, w = features.shape

        w_stride = h / self.output_size
        h_stride = w / self.output_size

        res = torch.zeros((num_channels, self.output_size, self.output_size))
        res_idx = torch.zeros((num_channels, self.output_size, self.output_size))

        for i in range(self.output_size):
            for j in range(self.output_size):

                w_start = int(np.floor(j * w_stride))
                w_end   = int(np.ceil((j + 1) * w_stride))

                h_start = int(np.floor(i * h_stride))
                h_end   = int(np.ceil((i + 1) * h_stride))

                w_start = np.clip(w_start, 0, w)
                w_end   = np.clip(w_end, 0, w)

                h_start = np.clip(h_start, 0, h)
                h_end   = np.clip(h_end, 0, h)

                patch = features[ :, h_start : h_end, w_start : w_end]
                max_value, max_idx = torch.max(patch.reshape(num_channels, -1), dim = 1)
                res[ :, i, j] = max_value
                res_idx[ :, i, j] = max_idx

        return res, res_idx

    def __call__(self, feature_layer, proposals): 

        batch_size, num_channels, _, _, = feature_layer.shape

        scaled_proposals = torch.zeros_like(proposals)

        scaled_proposals[ :, 0] = torch.ceil(proposals[ :, 0] * self.scaling_factor)
        scaled_proposals[ :, 1] = torch.ceil(proposals[ :, 1] * self.scaling_factor)
        scaled_proposals[ :, 2] = torch.ceil(proposals[ :, 2] * self.scaling_factor)
        scaled_proposals[ :, 3] = torch.ceil(proposals[ :, 3] * self.scaling_factor)

        res = torch.zeros((len(proposals), num_channels, self.output_size, self.output_size))
        res_idx = torch.zeros((len(proposals), num_channels, self.output_size, self.output_size))

        for idx in range(len(proposals)):
            proposal = scaled_proposals[idx]

            extracted_feat = feature_layer[0, :, proposal[1].to(dtype = torch.int8) : proposal[3].to(dtype = torch.int8) + 1, proposal[0].to(dtype = torch.int8) : proposal[2].to(dtype = torch.int8) + 1]
            res[idx], res_idx[idx] = self._roi_pool(extracted_feat)

        return res

device = torch.device('cpu')
# torch.set_default_tensor_type(torch.cpu.DoubleTensor)

# create feature layer, proposals and targets
num_proposals = 10
feat_layer = torch.randn(1, 64, 32, 32)

proposals = torch.zeros((num_proposals, 4))
proposals[:, 0] = torch.randint(0, 16, (num_proposals,))
proposals[:, 1] = torch.randint(0, 16, (num_proposals,))
proposals[:, 2] = torch.randint(16, 32, (num_proposals,))
proposals[:, 3] = torch.randint(16, 32, (num_proposals,))


my_roi_pool_obj = TorchROIPool(3, 2**-1)
roi_pool1 = my_roi_pool_obj(feat_layer, proposals)

roi_pool_obj = RoIPool(3, 2**-1)
roi_pool2 = roi_pool_obj(feat_layer, [proposals])

np.testing.assert_array_almost_equal(roi_pool1.reshape(-1,1), roi_pool2.reshape(-1,1))

my_roi_align_obj = TorchROIAlign(7, 2**-1)
roi_align1 = my_roi_align_obj(feat_layer, proposals)
roi_align1 = roi_align1.cpu().numpy()

roi_align_obj = RoIAlign(7, 2**-1, sampling_ratio=2, aligned=False)
roi_align2 = roi_align_obj(feat_layer, [proposals])
roi_align2 = roi_align2.cpu().numpy()

np.testing.assert_array_almost_equal(roi_align0.reshape(-1,1), roi_align2.reshape(-1,1))

import torch
import torch.nn as nn

class RegionProposalNetwork(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, num_anchor):
        super().__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.num_anchor   = num_anchor
        self.kernel_size  = kernel_size

        self.intermediate = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, stride = 1, padding = 1)

        self.classification_head = nn.Conv2d(self.out_channels, self.num_anchor, 1)
        self.regression_head = nn.Conv2d(self.out_channels, 4 * self.num_anchor, 1)

        for layer in self.children():
            nn.init.normal_(layer.weight, std = 0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, feature_map):
        t = nn.functional.relu(self.intermediate(feature_map))
        classification_op = self.classification_head(t)
        regression_op = self.regression_head(t)

        classification_op = classification_op.permute(0, 2, 3, 1).flatten()
        regression_op = regression_op.permute(0, 2, 3, 1).reshape(-1, 4) 

        return classification_op, regression_op

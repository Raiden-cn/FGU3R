import torch.nn as nn


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        # self.project_1 = nn.Sequential(
        #     nn.Conv2d(320, self.num_bev_features, 1),
        #     # partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)(self.num_bev_features)
        #     nn.BatchNorm2d(self.num_bev_features),
        #     nn.ReLU(),
        # )
        
    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:
        """

        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        # if C * D != self.num_bev_features:
        #     spatial_features = self.project_1(spatial_features)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict
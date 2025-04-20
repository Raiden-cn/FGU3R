import torch.nn as nn
import torch
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
import torch.nn.functional as F
from ...utils import common_utils, spconv_utils, loss_utils
from .roi_head_template import RoIHeadTemplate
from .fgu3r_utils import CAAF

class FGU3RHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, point_cloud_range=None, voxel_size=None, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.roi_grid_pool_layer, num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
            input_channels=input_channels['PFE'],
            config=self.model_cfg.ROI_GRID_POOL
        )
        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * num_c_out

        self.pool_cfg_mm = model_cfg.ROI_GRID_POOL_MM
        LAYER_cfg_mm = self.pool_cfg_mm.POOL_LAYERS
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        
        c_out_mm = 0
        self.roi_grid_pool_layers_mm = nn.ModuleList()
        feat = self.pool_cfg_mm.get('FEAT_NUM', 1)
        for src_name in self.pool_cfg_mm.FEATURES_SOURCE:
            mlps = LAYER_cfg_mm[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [input_channels[src_name]*feat] + mlps[k]
            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                query_ranges=LAYER_cfg_mm[src_name].QUERY_RANGES,
                nsamples=LAYER_cfg_mm[src_name].NSAMPLE,
                radii=LAYER_cfg_mm[src_name].POOL_RADIUS,
                mlps=mlps,
                pool_method=LAYER_cfg_mm[src_name].POOL_METHOD,
            )

            self.roi_grid_pool_layers_mm.append(pool_layer)
            c_out_mm += sum([x[-1] for x in mlps])

        # main
        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                # nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.Conv1d(pre_channel * 2 if k == 0 else pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )

        if self.training:
            # mm
            shared_fc_list_pseudo = []
            pre_channel_mm = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out_mm
            for k in range(0, self.model_cfg.SHARED_FC_MM.__len__()):
                shared_fc_list_pseudo.extend([
                    nn.Conv1d(pre_channel_mm, self.model_cfg.SHARED_FC_MM[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(self.model_cfg.SHARED_FC_MM[k]),
                    nn.ReLU()
                ])
                pre_channel_mm = self.model_cfg.SHARED_FC_MM[k]
            self.shared_fc_layer_mm = nn.Sequential(*shared_fc_list_pseudo)
            self.cls_pred_layer_mm = self.make_fc_layers(
                input_channels=pre_channel_mm, 
                output_channels=self.num_class, 
                fc_list=self.model_cfg.CLS_FC_MM
            )
            self.reg_pred_layer_mm = self.make_fc_layers(
                input_channels=pre_channel_mm,
                output_channels=self.box_coder.code_size * self.num_class,
                fc_list=self.model_cfg.REG_FC_MM
            )

            # raw
            shared_fc_list_raw = []
            pre_channel_raw = GRID_SIZE * GRID_SIZE * GRID_SIZE * num_c_out
            for k in range(0, self.model_cfg.SHARED_FC_RAW.__len__()):
                shared_fc_list_raw.extend([
                    nn.Conv1d(pre_channel_raw, self.model_cfg.SHARED_FC_RAW[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(self.model_cfg.SHARED_FC_RAW[k]),
                    nn.ReLU()
                ])
                pre_channel_raw = self.model_cfg.SHARED_FC_RAW[k]
            self.shared_fc_layer_raw = nn.Sequential(*shared_fc_list_raw)

            self.cls_pred_layer_raw = self.make_fc_layers(
                input_channels=pre_channel_raw, 
                output_channels=self.num_class, 
                fc_list=self.model_cfg.CLS_FC_RAW
            )
            self.reg_pred_layer_raw = self.make_fc_layers(
                input_channels=pre_channel_raw,
                output_channels=self.box_coder.code_size * self.num_class,
                fc_list=self.model_cfg.REG_FC_RAW
            )


        # CAAFusion
        self.fusion = CAAF(num_c_out, c_out_mm, self.model_cfg.ATTENTION_FC[0])

        self.init_weights(weight_init='xavier')
        self.build_losses_mm(self.model_cfg.LOSS_CONFIG)
        self.build_losses_raw(self.model_cfg.LOSS_CONFIG)

    def build_losses_mm(self, losses_cfg):
        self.add_module(
            'reg_loss_func_mm',
            loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS_MM['code_weights'])
        )

    def build_losses_raw(self, losses_cfg):
        self.add_module(
            'reg_loss_func_raw',
            loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS_RAW['code_weights'])
        )

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.reg_pred_layer_mm[-1].weight, mean=0, std=0.001)
        nn.init.normal_(self.reg_pred_layer_raw[-1].weight, mean=0, std=0.001)


    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']

        point_features = point_features * batch_dict['point_cls_scores'].view(-1, 1)

        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)  # (B, Nx6x6x6, 3)

        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        new_xyz = global_roi_grid_points.view(-1, 3)
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])
        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),
        )  # (M1 + M2 ..., C)


        pooled_features = pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            pooled_features.shape[-1]
        )  # (BxN, 6x6x6, C)
        return pooled_features

    
    def roi_grid_pool_mm(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        rois = batch_dict['rois'].clone()
        #rois[:, 3:5] = rois[:, 3:5]*0.5

        batch_size = batch_dict['batch_size']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)

        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.model_cfg.ROI_GRID_POOL_MM.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        # roi_grid_xyz: (B, Nx6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)

        # compute the voxel coordinates of grid points
        roi_grid_coords_x = (roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) // self.voxel_size[0]
        roi_grid_coords_y = (roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) // self.voxel_size[1]
        roi_grid_coords_z = (roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) // self.voxel_size[2]
        # roi_grid_coords: (B, Nx6x6x6, 3)
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)

        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        # roi_grid_coords: (B, Nx6x6x6, 4)
        # roi_grid_coords = torch.cat([batch_idx, roi_grid_coords], dim=-1)
        # roi_grid_coords = roi_grid_coords.int()
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1])

        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg_mm.FEATURES_SOURCE):
            pool_layer = self.roi_grid_pool_layers_mm[k]
            if src_name in ['x_conv1', 'x_conv2', 'x_conv3', 'x_conv4']:

                cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
                # while 'multi_scale_3d_features_mm'+rot_num_id not in batch_dict:
                #     j-=1
                #     rot_num_id = str(j)
                cur_sp_tensors = batch_dict['multi_scale_3d_features_mm'][src_name]

                # if with_vf_transform:
                #     cur_sp_tensors = batch_dict['multi_scale_3d_features_post'][src_name]
                # else:
                #     cur_sp_tensors = batch_dict['multi_scale_3d_features_mm'+rot_num_id][src_name]

                # compute voxel center xyz and batch_cnt
                cur_coords = cur_sp_tensors.indices
                cur_voxel_xyz = common_utils.get_voxel_centers(
                    cur_coords[:, 1:4],
                    downsample_times=cur_stride,
                    voxel_size=self.voxel_size,
                    point_cloud_range=self.point_cloud_range
                )  #
                cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
                for bs_idx in range(batch_size):
                    cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
                # get voxel2point tensor

                v2p_ind_tensor = spconv_utils.generate_voxel2pinds(cur_sp_tensors)

                # compute the grid coordinates in this scale, in [batch_idx, x y z] order
                cur_roi_grid_coords = roi_grid_coords // cur_stride
                cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
                cur_roi_grid_coords = cur_roi_grid_coords.int()
                # voxel neighbor aggregation
                pooled_features = pool_layer(
                    xyz=cur_voxel_xyz.contiguous(),
                    xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                    new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                    new_xyz_batch_cnt=roi_grid_batch_cnt,
                    new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                    features=cur_sp_tensors.features.contiguous(),
                    voxel2point_indices=v2p_ind_tensor
                )

                pooled_features = pooled_features.view(
                    -1, self.pool_cfg_mm.GRID_SIZE ** 3,
                    pooled_features.shape[-1]
                )  # (BxN, 6x6x6, C)
                pooled_features_list.append(pooled_features)

        ms_pooled_features = torch.cat(pooled_features_list, dim=-1)

        return ms_pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = batch_dict.get('roi_targets_dict', None)
            if targets_dict is None:
                targets_dict = self.assign_targets(batch_dict)
                batch_dict['rois'] = targets_dict['rois']
                batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        pooled_features_raw = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        # TODO RoI pseudo pooling
        pooled_features_mm = self.roi_grid_pool_mm(batch_dict)  # (BxN, 6x6x6, C)
        
        # cat
        # final_pooled_features = torch.cat([pooled_features, pooled_features_mm], dim=-1)
        # TODO sum

        # TODO att
        pooled_features_raw = pooled_features_raw.transpose(1,2) # (B, C, N**3) --> (B, N**3, C)
        pooled_features_mm = pooled_features_mm.transpose(1,2) # (B, C, N**3) --> (B, N**3, C)

        final_pooled_features = self.fusion(pooled_features_raw, pooled_features_mm) # [512, C, N**3]

        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = final_pooled_features.shape[0]
        # v1 v2 wrong reshape
        # pooled_features = final_pooled_features.permute(0, 2, 1).\
        #     contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)
        # v3 channel in middle (BxN, C, 6, 6, 6)
        pooled_features = final_pooled_features.contiguous().\
            view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if self.training:
            x_mm = pooled_features_mm.reshape(pooled_features_mm.size(0), -1, 1)
            shared_features_mm = self.shared_fc_layer_mm(x_mm)
            rcnn_cls_mm = self.cls_pred_layer_mm(shared_features_mm)
            rcnn_reg_mm = self.reg_pred_layer_mm(shared_features_mm)

            x_raw = pooled_features_raw.reshape(pooled_features_raw.size(0), -1, 1)
            shared_features_raw = self.shared_fc_layer_raw(x_raw)
            rcnn_cls_raw = self.cls_pred_layer_raw(shared_features_raw)
            rcnn_reg_raw = self.reg_pred_layer_raw(shared_features_raw)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False

        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            targets_dict['rcnn_cls_mm'] = rcnn_cls_mm
            targets_dict['rcnn_reg_mm'] = rcnn_reg_mm
            targets_dict['rcnn_cls_raw'] = rcnn_cls_raw
            targets_dict['rcnn_reg_raw'] = rcnn_reg_raw
            self.forward_ret_dict = targets_dict

        return batch_dict



    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0

        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_reg
        tb_dict.update(reg_tb_dict)

        # pseudo
        if 'rcnn_cls_mm' in self.forward_ret_dict:
            rcnn_loss_cls_mm, cls_tb_dict_mm = self.get_box_cls_layer_loss_mm(self.forward_ret_dict)
            rcnn_loss += rcnn_loss_cls_mm
            tb_dict.update(cls_tb_dict_mm)
        if 'rcnn_reg_mm' in self.forward_ret_dict:
            rcnn_loss_reg_mm, reg_tb_dict_mm = self.get_box_reg_layer_loss_mm(self.forward_ret_dict)
            rcnn_loss += rcnn_loss_reg_mm
            tb_dict.update(reg_tb_dict_mm)

        # raw
        if 'rcnn_cls_raw' in self.forward_ret_dict:
            rcnn_loss_cls_raw, cls_tb_dict_raw = self.get_box_cls_layer_loss_raw(self.forward_ret_dict)
            rcnn_loss += rcnn_loss_cls_raw
            tb_dict.update(cls_tb_dict_raw)
        if 'rcnn_reg_raw' in self.forward_ret_dict:
            rcnn_loss_reg_raw, reg_tb_dict_raw = self.get_box_reg_layer_loss_raw(self.forward_ret_dict)
            rcnn_loss += rcnn_loss_reg_raw
            tb_dict.update(reg_tb_dict_raw)

        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict

    def get_box_cls_layer_loss_mm(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls_mm']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS_MM['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls_mm': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict

    def get_box_reg_layer_loss_mm(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg_mm']  # (rcnn_batch_size, C)
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}

        if loss_cfgs.REG_LOSS == 'smooth-l1':
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
            )

            rcnn_loss_reg = self.reg_loss_func_mm(
                rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                reg_targets.unsqueeze(dim=0),
            )  # [B, M, 7]
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS_MM['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg_mm'] = rcnn_loss_reg.item()

            # if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:
            #     # TODO: NEED to BE CHECK
            #     fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
            #     fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]

            #     fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
            #     batch_anchors = fg_roi_boxes3d.clone().detach()
            #     roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
            #     roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
            #     batch_anchors[:, :, 0:3] = 0
            #     rcnn_boxes3d = self.box_coder.decode_torch(
            #         fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
            #     ).view(-1, code_size)

            #     rcnn_boxes3d = common_utils.rotate_points_along_z(
            #         rcnn_boxes3d.unsqueeze(dim=1), roi_ry
            #     ).squeeze(dim=1)
            #     rcnn_boxes3d[:, 0:3] += roi_xyz

            #     loss_corner = loss_utils.get_corner_loss_lidar(
            #         rcnn_boxes3d[:, 0:7],
            #         gt_of_rois_src[fg_mask][:, 0:7]
            #     )
            #     loss_corner = loss_corner.mean()
            #     loss_corner = loss_corner * loss_cfgs.LOSS_WEIGHTS_MM['rcnn_corner_weight']

            #     rcnn_loss_reg += loss_corner
            #     tb_dict['rcnn_loss_corner'] = loss_corner.item()
        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict

    def get_box_cls_layer_loss_raw(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls_raw']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS_RAW['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls_raw': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict

    def get_box_reg_layer_loss_raw(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg_raw']  # (rcnn_batch_size, C)
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}

        if loss_cfgs.REG_LOSS == 'smooth-l1':
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
            )

            rcnn_loss_reg = self.reg_loss_func_raw(
                rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                reg_targets.unsqueeze(dim=0),
            )  # [B, M, 7]
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS_RAW['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg_raw'] = rcnn_loss_reg.item()

            # if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:
            #     # TODO: NEED to BE CHECK
            #     fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
            #     fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]

            #     fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
            #     batch_anchors = fg_roi_boxes3d.clone().detach()
            #     roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
            #     roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
            #     batch_anchors[:, :, 0:3] = 0
            #     rcnn_boxes3d = self.box_coder.decode_torch(
            #         fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
            #     ).view(-1, code_size)

            #     rcnn_boxes3d = common_utils.rotate_points_along_z(
            #         rcnn_boxes3d.unsqueeze(dim=1), roi_ry
            #     ).squeeze(dim=1)
            #     rcnn_boxes3d[:, 0:3] += roi_xyz

            #     loss_corner = loss_utils.get_corner_loss_lidar(
            #         rcnn_boxes3d[:, 0:7],
            #         gt_of_rois_src[fg_mask][:, 0:7]
            #     )
            #     loss_corner = loss_corner.mean()
            #     loss_corner = loss_corner * loss_cfgs.LOSS_WEIGHTS_MM['rcnn_corner_weight']

            #     rcnn_loss_reg += loss_corner
            #     tb_dict['rcnn_loss_corner'] = loss_corner.item()
        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict
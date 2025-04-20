import torch
import torch.nn as nn

# TODO 

class PseudoCONV(nn.module):


    def __init__(self):
        super().__init__()


    def roicrop3d_gpu(self, batch_dict, pool_extra_width):
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
        batch_size     = batch_dict['batch_size']
        rois           = batch_dict['rois']
        num_rois       = rois.shape[1]

        enlarged_rois = box_utils.enlarge_box3d(rois.view(-1, 7).clone().detach(), pool_extra_width).view(batch_size, -1, 7) 
        batch_idx      = batch_dict['points_pseudo'][:, 0]
        point_coords   = batch_dict['points_pseudo'][:, 1:4]
        point_features = batch_dict['points_pseudo'][:,1:]    # N, 8{x,y,z,r,g,b,u,v}                          

        point_depths = point_coords.norm(dim=1) / self.model_cfg.ROI_POINT_CROP.DEPTH_NORMALIZER - 0.5
        point_features_list = [point_features, point_depths[:, None]]
        point_features = torch.cat(point_features_list, dim=1)   
        w, h = 1400, 400

        with torch.no_grad():
            total_pts_roi_index = []
            total_pts_batch_index = []
            total_pts_features = []
            for bs_idx in range(batch_size):
                bs_mask          = (batch_idx == bs_idx)
                cur_point_coords = point_coords[bs_mask]
                cur_features     = point_features[bs_mask]
                cur_roi          = enlarged_rois[bs_idx][:, 0:7].contiguous()

                box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(       
                    cur_point_coords.unsqueeze(0), cur_roi.unsqueeze(0)
                )      
                cur_box_idxs_of_pts = box_idxs_of_pts[0]

                points_in_rois = cur_box_idxs_of_pts != -1
                cur_box_idxs_of_pts = cur_box_idxs_of_pts[points_in_rois] + num_rois * bs_idx

                cur_pts_batch_index = cur_box_idxs_of_pts.new_zeros((cur_box_idxs_of_pts.shape[0]))
                cur_pts_batch_index[:] = bs_idx

                cur_features        = cur_features[points_in_rois]

                total_pts_roi_index.append(cur_box_idxs_of_pts)
                total_pts_batch_index.append(cur_pts_batch_index)
                total_pts_features.append(cur_features)

            total_pts_roi_index     =  torch.cat(total_pts_roi_index, dim=0)
            total_pts_batch_index =  torch.cat(total_pts_batch_index, dim=0)
            total_pts_features      =  torch.cat(total_pts_features, dim=0)
            total_pts_features_xyz_src = total_pts_features.clone()[...,:3]
            total_pts_rois = torch.index_select(rois.view(-1,7), 0, total_pts_roi_index.long())

            total_pts_features[:, 0:3] -= total_pts_rois[:, 0:3]
            total_pts_features[:, 0:3] = common_utils.rotate_points_along_z(
                total_pts_features[:, 0:3].unsqueeze(dim=1), -total_pts_rois[:, 6]
            ).squeeze(dim=1)          
            total_pts_features_raw = total_pts_features.clone()
            global_dv = total_pts_roi_index * h 
            total_pts_features[:, 7] += global_dv

        image = total_pts_features.new_zeros((batch_size*num_rois*h, w)).long()  
        global_index = torch.arange(1, total_pts_features.shape[0]+1)
        image[total_pts_features[:,7].long(), total_pts_features[:,6].long()] = global_index.to(device=total_pts_features.device)

        coords = getattr(self, self.model_cfg.ROI_AWARE_POOL.KERNEL_TYPE)
        points_list = []
        for circle_i in range(len(coords)):
            dx, dy = coords[circle_i]
            points_cur = image[total_pts_features[:, 7].long() + dx, total_pts_features[:, 6].long() + dy]
            points_list.append(points_cur)
        total_pts_neighbor = torch.stack(points_list,dim=0).transpose(0,1).contiguous()

        zero_features = total_pts_features.new_zeros((1,total_pts_features.shape[-1]))
        total_pts_features = torch.cat([zero_features,total_pts_features],dim=0)
        zero_neighbor = total_pts_neighbor.new_zeros((1,total_pts_neighbor.shape[-1]))
        total_pts_neighbor = torch.cat([zero_neighbor,total_pts_neighbor],dim=0)
        total_pts_batch_index = total_pts_batch_index.float().unsqueeze(dim=-1)

        return total_pts_features, total_pts_neighbor, total_pts_batch_index, total_pts_roi_index, total_pts_features_xyz_src
        
    def forward(self, batch_dict):

        
        return None
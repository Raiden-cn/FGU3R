import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, channels):
        super(Attention, self).__init__()
        self.pseudo_in, self.valid_in = channels
        middle = self.valid_in // 4
        self.fc1 = nn.Linear(self.pseudo_in, middle)
        self.fc2 = nn.Linear(self.valid_in, middle)
        self.fc3 = nn.Linear(2 * middle, 2)
        self.conv1 = nn.Sequential(nn.Conv1d(self.pseudo_in, self.valid_in, 1),
                                    nn.BatchNorm1d(self.valid_in),
                                    nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(self.valid_in, self.valid_in, 1),
                                    nn.BatchNorm1d(self.valid_in),
                                    nn.ReLU())

    def forward(self, pseudo_feas, valid_feas):
        batch = pseudo_feas.size(0)

        pseudo_feas_f = pseudo_feas.transpose(1,2).contiguous().view(-1, self.pseudo_in) # (b, grid, c).view(-1, c)
        valid_feas_f = valid_feas.transpose(1,2).contiguous().view(-1, self.valid_in) 

        pseudo_feas_f_ = self.fc1(pseudo_feas_f)
        valid_feas_f_ = self.fc2(valid_feas_f)
        pseudo_valid_feas_f = torch.cat([pseudo_feas_f_, valid_feas_f_], dim=-1)
        weight = torch.sigmoid(self.fc3(pseudo_valid_feas_f))

        pseudo_weight = weight[:,0].squeeze()
        pseudo_weight = pseudo_weight.view(batch, 1, -1)

        valid_weight = weight[:,1].squeeze()
        valid_weight = valid_weight.view(batch, 1, -1)

        pseudo_features_att = self.conv1(pseudo_feas) * pseudo_weight
        valid_features_att = self.conv2(valid_feas) * valid_weight

        return pseudo_features_att, valid_features_att

class CAAF(nn.Module):
    def __init__(self, pseudo_in, valid_in, outplanes):
        super(CAAF, self).__init__()
        self.attention = Attention(channels = [pseudo_in, valid_in])
        self.conv1 = torch.nn.Conv1d(pseudo_in + valid_in, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)

    def forward(self, pseudo_features, valid_features):
        pseudo_features_att, valid_features_att = self.attention(pseudo_features, valid_features)
        fusion_features = torch.cat([valid_features_att, pseudo_features_att], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features



import torch
from torch import nn
from torch.nn import functional as F

import inversion.configs.paths_config as path_config
from inversion.psp.encoders.model_irse import Backbone

from inversion.utils.debugger import set_trace


class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(path_config.ir_se50))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()


    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        # x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def old_forward(self, y_hat, y, x):
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        id_logs = []
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            diff_input = y_hat_feats[i].dot(x_feats[i])
            diff_views = y_feats[i].dot(x_feats[i])
            id_logs.append({'diff_target': float(diff_target),
                            'diff_input': float(diff_input),
                            'diff_views': float(diff_views)})
            loss += 1 - diff_target
            id_diff = float(diff_target) - float(diff_views)
            sim_improvement += id_diff
            count += 1

        return loss / count, sim_improvement / count, id_logs

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        # x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        # sim_improvement = 0
        # id_logs = []
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            # diff_input = y_hat_feats[i].dot(x_feats[i])
            # diff_views = y_feats[i].dot(x_feats[i])
            # id_logs.append({'diff_target': float(diff_target),
            #                 'diff_input': float(diff_input),
            #                 'diff_views': float(diff_views)})
            loss += 1 - diff_target
            # id_diff = float(diff_target) - float(diff_views)
            # sim_improvement += id_diff
            count += 1

        return loss / count

    def forward_triplet_loss(self, anchor, positive, negative, tgt_img, src_img):
        n_samples = anchor.shape[0]
        a_feats = self.extract_feats(anchor)
        p_feats = self.extract_feats(positive)
        n_feats = self.extract_feats(negative)

        # Margin
        src_feats = self.extract_feats(src_img)
        tgt_feats = self.extract_feats(tgt_img)

        loss = 0
        count = 0
        for i in range(n_samples):
            margin = tgt_feats[i].dot(src_feats[i])
            loss += F.triplet_margin_loss(anchor=a_feats, positive=p_feats,
                                          negative=n_feats, margin=margin.item())
            count += 1

        return loss / count

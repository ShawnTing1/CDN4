import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry
from modules.utils import batched_index_select

from modules.utils import _l2norm
from .similarity import Similarity


@registry.Query.register("CDN4")
class CDN4(nn.Module):
    
    def __init__(self, in_channels, cfg):
        super().__init__()

        self.cfg = cfg
        self.neighbor_k = 1

        self.inner_simi = Similarity(cfg, metric='cosine')
        # self.inner_simi = Similarity(cfg, metric='innerproduct')
        # self.inner_simi = Similarity(cfg, metric='euclidean')
        # self.inner_simi = Similarity(cfg, metric='manhattan')
        # self.inner_simi = Similarity(cfg, metric='chebyshev')
        self.criterion = nn.CrossEntropyLoss()
        self.k_shot_average = cfg.model.dn4.larger_shot == "average"
        if cfg.model.encoder == "R12":
            self.channel = 640
        else:
            self.channel = 64


    def forward(self, support_xf, trans_support_xf, support_y, query_xf, trans_query_xf, query_y, n_way, k_shot):

        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]

        # If the number of shots is larger than 1, we need to average the features
        if self.k_shot_average:
            support_xf = support_xf.view(b, n_way, k_shot, c, h, w).mean(2)
            trans_support_xf = trans_support_xf.view(b, n_way, k_shot, c, h, w).mean(2)
            support_xf = support_xf.view(b, n_way, c, h * w)
            trans_support_xf = trans_support_xf.view(b, n_way, c, h * w)

        # support_xf [b, n_way, c, h, w]

        # Given the original feature and transformed feature, calculate the similarity matrix
        support_xf = support_xf.view(support_xf.size(0), support_xf.size(1), support_xf.size(2), -1) # [b, n_way, c, M_s], M_s = h * w
        trans_support_xf = trans_support_xf.view(support_xf.size(0), support_xf.size(1), support_xf.size(2), -1)

        # Calculate the similarity matrix       
        innerproduct_matrix_1 = self.inner_simi(support_xf, query_xf) # [b, q, n_way, M_s, M_q], M_s = h * w, M_q = h * w
        innerproduct_matrix_2 = self.inner_simi(trans_support_xf, trans_query_xf)
        innerproduct_matrix_cross_1 = self.inner_simi(support_xf, trans_query_xf)
        innerproduct_matrix_cross_2 = self.inner_simi(trans_support_xf, query_xf)

        # Take topk for the last dimension, neighbor_k=1
        topk_value_1, _ = torch.topk(innerproduct_matrix_1, self.neighbor_k, -1) # [b, q, n_way, M_q, neighbor_k]
        topk_value_2, _ = torch.topk(innerproduct_matrix_2, self.neighbor_k, -1) # [b, q, n_way, M_q, neighbor_k]
        topk_value_cross_1, _ = torch.topk(innerproduct_matrix_cross_1, self.neighbor_k, -1) # [b, q, n_way, M_q, neighbor_k]
        topk_value_cross_2, _ = torch.topk(innerproduct_matrix_cross_2, self.neighbor_k, -1) # [b, q, n_way, M_q, neighbor_k]

        # Calculate the mean for M_q
        similarity_matrix_1 = topk_value_1.mean(-1).view(b, q, n_way, -1).sum(-1) # [b, q, n_way]
        similarity_matrix_2 = topk_value_2.mean(-1).view(b, q, n_way, -1).sum(-1)
        similarity_matrix_cross_1 = topk_value_cross_1.mean(-1).view(b, q, n_way, -1).sum(-1)
        similarity_matrix_cross_2 = topk_value_cross_2.mean(-1).view(b, q, n_way, -1).sum(-1)
        
        similarity_matrix_1 = similarity_matrix_1.view(b * q, n_way) # [b * q, n_way]
        similarity_matrix_2 = similarity_matrix_2.view(b * q, n_way)
        similarity_matrix_cross_1 = similarity_matrix_cross_1.view(b * q, n_way)
        similarity_matrix_cross_2 = similarity_matrix_cross_2.view(b * q, n_way)

        query_y = query_y.view(b * q)
        if self.training:
            loss_1 = self.criterion(similarity_matrix_1, query_y)
            loss_2 = self.criterion(similarity_matrix_cross_1, query_y)
            loss_3 = self.criterion(similarity_matrix_cross_2, query_y)
            loss_4 = self.criterion(similarity_matrix_2, query_y)

            loss = (loss_1 + loss_2 + loss_3 + loss_4) / 4

            return {"dn4_loss": loss}
        else:
            similarity_matrix = (similarity_matrix_1 + similarity_matrix_2 + similarity_matrix_cross_1 + similarity_matrix_cross_2) / 4
            _, predict_labels = torch.max(similarity_matrix, 1)
            rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(len(query_y))]
            return rewards

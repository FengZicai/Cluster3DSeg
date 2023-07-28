import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import time 
from torch_scatter import scatter_mean, scatter_max
import math

def online_clustering(self):
    N = self.PS.size(0)
    self.PS = self.PS.T.double() # now it is K x N
    r = torch.ones((self.K, 1), dtype=self.dtype).to(self.dev) / self.K
    c = torch.ones((N, 1), dtype=self.dtype).to(self.dev) / N
    ones = torch.ones(N, dtype=self.dtype).to(self.dev)

    self.PS = self.PS.pow(self.lamb)  # K x N
    inv_K = 1. / self.K
    inv_N = 1. / N

    err = 1e6
    _counter = 0
    while err > 1e-1:
        r = inv_K / (self.PS @ c)          # (KxN)@(N,1) = K x 1
        c_new = inv_N / (r.T @ self.PS).T  # ((1,K)@(KxN)).t() = N x 1
        if _counter % 10 == 0:
            err = torch.abs((c.squeeze() / c_new.squeeze()) - ones)
            err = torch.sum(torch.where(torch.isnan(err), torch.full_like(err, 0), err))

        c = c_new
        _counter += 1
    self.PS *= torch.squeeze(c)
    self.PS = self.PS.T
    self.PS *= torch.squeeze(r)
    self.PS = self.PS.T
    argmaxes = torch.argmax(self.PS, dim=0, keepdim=False) # size N
    newL = argmaxes.long()
    self.L = newL.to(self.dev)

    self.PS = 0
    

class clusterContrastLoss(nn.Module):
    def __init__(self, ignore_label, device, k=40, mu=0.9999):
        super(clusterContrastLoss, self).__init__()

        self.temperature = 0.1
        self.base_temperature = 2
        self.ignore_label = ignore_label
        self.max_samples = 100
        self.max_views = 100

        # Sinkhorn-Knopp stuff
        self.K = k #number of clusters
        self.k_ban = 10 
        self.k_ban_class = [7,8]

        # initiate labels as shuffled.
        self.dev = device
        self.dtype = torch.float64

        self.lamb = 25  # the parameter lambda in the SK algorithm

        # MEM stuff
        self.num_classes = 19
        self.dim = 64
        self.mu = mu

        # cluster_center
        self.cluster_center = torch.randn((self.num_classes, self.K, self.dim),requires_grad=False).to(self.dev)
        self.cluster_center = nn.functional.normalize(self.cluster_center, p=2, dim=2)

        self.pixel_update_freq = 10 # number  V , of pixels
        self.pixel_size = self.pixel_update_freq * 5

        self.point_queue = torch.randn((self.num_classes*self.K, self.pixel_size, self.dim),requires_grad=False).to(self.dev)
        self.point_queue = nn.functional.normalize(self.point_queue, p=2, dim=2)
        self.point_queue_ptr = torch.zeros(self.num_classes*self.K, dtype=torch.long,requires_grad=False).to(self.dev)

    def _update_operations(self):
        self.cluster_center = self.cluster_center * self.mu + self.new_cluster_center * (1 - self.mu)
        self.cluster_center = nn.functional.normalize(self.cluster_center, p=2, dim=2).detach_()

    def _queue_operations(self, feats, labels):

        this_feat = feats.contiguous().view(self.dim, -1)
        this_label = labels.contiguous().view(-1)
        this_label_ids = torch.unique(this_label)
        this_label_ids = [x for x in this_label_ids if (x > 0) and (x != self.ignore_label)]

        for lb in this_label_ids:
            idxs = (this_label == lb).nonzero(as_tuple=False)

            # pixel enqueue and dequeue
            num_pixel = idxs.shape[0]
            perm = torch.randperm(num_pixel)
            updata_cnt = min(num_pixel, self.pixel_update_freq)
            feat = this_feat[:, perm[:updata_cnt]]
            feat = torch.transpose(feat, 0, 1)
            ptr = int(self.point_queue_ptr[lb])

            if ptr + updata_cnt > self.pixel_size:
                self.point_queue[lb, -updata_cnt:, :] = nn.functional.normalize(feat, p=2, dim=1).detach_()
                self.point_queue_ptr[lb] = 0
            else:
                self.point_queue[lb, ptr:ptr + updata_cnt, :] = nn.functional.normalize(feat, p=2, dim=1).detach_()
                self.point_queue_ptr[lb] = (self.point_queue_ptr[lb] + updata_cnt) % self.pixel_size


    def _assigning_subclass_labels(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero(as_tuple=False).shape[0]]

            classes.append(this_classes)
            total_classes += len(this_classes)

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = []
        y_ = []
        X_contrast = []
        y_contrast = []

        self.new_cluster_center = torch.zeros((self.num_classes, self.K, self.dim)).to(self.dev).detach()

        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_x = X[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero(as_tuple=False)
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero(as_tuple=False)

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                    perm = torch.randperm(num_hard)
                    hard_indices = hard_indices[perm[:num_hard_keep]]
                    perm = torch.randperm(num_easy)
                    easy_indices = easy_indices[perm[:num_easy_keep]]
                    indices = torch.cat((hard_indices, easy_indices), dim=0).squeeze(1)
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                    perm = torch.randperm(num_hard)
                    hard_indices = hard_indices[perm[:num_hard_keep]]
                    perm = torch.randperm(num_easy)
                    easy_indices = easy_indices[perm[:num_easy_keep]]
                    indices = torch.cat((hard_indices, easy_indices), dim=0).squeeze(1)
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                    perm = torch.randperm(num_hard)
                    hard_indices = hard_indices[perm[:num_hard_keep]]
                    perm = torch.randperm(num_easy)
                    easy_indices = easy_indices[perm[:num_easy_keep]]
                    indices = torch.cat((hard_indices, easy_indices), dim=0).squeeze(1)
                else:
                    indices = torch.cat((hard_indices, easy_indices), dim=0).squeeze(1)

                if cls_id in self.k_ban_class:
                    pc = self.cluster_center[cls_id][:self.k_ban]
                    xc = this_x[indices]
                    yc = this_y_hat[indices]
                    self.PS = torch.mm(xc, pc.T)
                    self.PS = nn.functional.softmax(self.PS, 1)
                    online_clustering(self)
                    yc = yc * self.K
                    yc = yc +  self.L

                    self.new_cluster_center[cls_id][:self.k_ban] = scatter_mean(xc, self.L, dim=0, dim_size=self.k_ban).detach()

                    X_.append(xc)
                    y_.append(yc)

                    X_contrast.append(pc)
                    y_contrast.append(cls_id.repeat(self.k_ban) * self.K + torch.tensor(list(range(self.k_ban))).long().to(self.dev))

                else:  
                    pc = self.cluster_center[cls_id]
                    xc = this_x[indices]
                    yc = this_y_hat[indices]
                    self.PS = torch.mm(xc, pc.T)
                    self.PS = nn.functional.softmax(self.PS, 1)
                    online_clustering(self)
                    yc = yc * self.K
                    yc = yc +  self.L

                    self.new_cluster_center[cls_id] = scatter_mean(xc, self.L, dim=0, dim_size=self.K).detach()

                    X_.append(xc)
                    y_.append(yc)

                    X_contrast.append(pc)
                    y_contrast.append(cls_id.repeat(self.K) * self.K + torch.tensor(list(range(self.K))).long().to(self.dev))
            
        X_ = torch.cat(X_,dim=0).float()
        y_ = torch.cat(y_,dim=0).float()

        X_contrast = torch.cat(X_contrast,dim=0).float()
        y_contrast = torch.cat(y_contrast,dim=0).float()

        return X_, y_, X_contrast, y_contrast

    def _sample_negative(self):
        class_num, cache_size, feat_size = self.point_queue.shape
        reduce_num = (self.K - self.k_ban) * len(self.k_ban_class)
        X_ = torch.zeros(((class_num - reduce_num) * cache_size , feat_size)).float().to(self.dev)
        y_ = torch.zeros(((class_num - reduce_num) * cache_size , 1)).float().to(self.dev)
        sample_ptr = 0
        for ii in range(class_num):
            if ii in range(7 * self.K + self.k_ban, 8 * self.K):
                continue
            if ii in range(8 * self.K + self.k_ban, 9 * self.K):
                continue
            this_q = self.point_queue[ii, :cache_size, :]
            X_[sample_ptr:sample_ptr + cache_size, ...] = this_q
            y_[sample_ptr:sample_ptr + cache_size, ...] = ii
            sample_ptr += cache_size

        return X_, y_

    def _ppc2_contrastive(self, X_anchor, y_anchor):
        
        y_anchor = y_anchor.contiguous().view(-1, 1)
        anchor_feature = X_anchor

        X_contrast, y_contrast = self._sample_negative()
        y_contrast = y_contrast.contiguous().view(-1, 1)

        contrast_feature = X_contrast
        contrast_label = y_contrast

        mask = torch.eq(y_anchor, contrast_label.T).float().to(self.dev) 

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # it is to avoid the numerical overflow

        neg_mask = 1 - mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True) #neg_logits denotes the sum of logits of all negative pairs of one anchor

        exp_logits = torch.exp(logits) # exp_logits denotes the logit of each sample pair

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def _ppc_contrastive(self, X_anchor, y_anchor):
        y_anchor = y_anchor.contiguous().view(-1, 1)
        anchor_feature = X_anchor
        anchor_num = X_anchor.shape[0]

        contrast_feature = X_anchor
        contrast_label = y_anchor

        mask = torch.eq(y_anchor, contrast_label.T).float().to(self.dev) 

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                        self.temperature)
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # it is to avoid the numerical overflow

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask
        neg_mask = 1 - mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True) #neg_logits denotes the sum of logits of all negative pairs of one anchor

        exp_logits = torch.exp(logits) # exp_logits denotes the logit of each sample pair

        log_prob = logits - torch.log(exp_logits + neg_logits)
        o_i = torch.where(mask.sum(1)!=0)[0]

        mean_log_prob_pos = (mask * log_prob).sum(1)[o_i] / mask.sum(1)[o_i]

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def _pcc_contrastive(self, X_anchor, y_anchor, X_contrast, y_contrast):
        y_anchor = y_anchor.contiguous().view(-1, 1)
        y_contrast = y_contrast.contiguous().view(-1, 1)

        anchor_feature = X_anchor 
        anchor_label = y_anchor

        contrast_feature = X_contrast
        contrast_label = y_contrast

        mask = torch.eq(anchor_label, contrast_label.T).float().to(self.dev) 

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # it is to avoid the numerical overflow

        neg_mask = 1 - mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True) #neg_logits denotes the sum of logits of all negative pairs of one anchor

        exp_logits = torch.exp(logits) # exp_logits denotes the logit of each sample pair

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None):
        labels = labels.long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])
        feats = nn.functional.normalize(feats, p=2, dim=2)

        feats_, labels_, feats_contrast, labels_contrast = self._assigning_subclass_labels(feats, labels, predict)

        loss = self._ppc2_contrastive(feats_, labels_)
        loss += self._ppc_contrastive(feats_, labels_)
        loss += self._pcc_contrastive(feats_, labels_, feats_contrast, labels_contrast)

        self._queue_operations(feats_, labels_.long())
        self._update_operations()
        return loss * 0.5

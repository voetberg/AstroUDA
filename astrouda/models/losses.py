from abc import ABC, abstractmethod

from astrouda.utils import utils
import torch 
import torch.nn.functional as F

class Loss(ABC): 
    def __init__(self, config):
        super().__init__()
        self.loss_fn = None
        self.config = config

    @abstractmethod
    def forward(self, inputs, targets, model=None):
        raise NotImplementedError
    
    def __call__(self, inputs, targets, model=None):
        return self.forward(inputs, targets, model)


class ObjectiveLoss(Loss): 
    """Objective loss such that L = L_CE + lambda*(L_AC + L_ES)"""
    def __init__(self, config): 
        super().__init__(config)
        self._lambda = config.get('lambda', 0.005)

        self.loss_fn_ce = CrossEntropyLoss(config)
        self.loss_fn_ac = AdaptiveClusteringLoss(config)
        self.loss_fn_es = EntropySeparationLoss(config)

    def forward(self, probabilities, labels, model):

        return (self.loss_fn_ce(probabilities, labels) + 
                self._lambda * (self.loss_fn_ac(probabilities, labels) + 
                                self.loss_fn_es(probabilities, labels)))


class CrossEntropyLoss(Loss):
    def __init__(self, config):
        super().__init__(config)
        self.reduction = config.get("reduction", "mean")
        # Pulled from the datasets - NS/(K*n_k), NS = images in source data, K = N source classes, n_k = images in class k

    def forward(self, inputs, targets):
        loss_fn_ce = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        return loss_fn_ce(inputs.to(torch.float32), targets.to(torch.float32))


class AdaptiveClusteringLoss(Loss):
    '''Adaptive clustering loss via https://openaccess.thecvf.com/content/CVPR2021/supplemental/Li_Cross-Domain_Adaptive_Clustering_CVPR_2021_supplemental.pdf
    
    Reference: https://github.com/lijichang/CVPR2021-SSDA/blob/main/losses.py

    Produce a pairwise similarity of similar classes across domains
    '''
    def __init__(self, config):
        super().__init__(config)
        self.top_k = config.get('top_k', 7)  # Pick 3 for 3 class problems, 7 for 10 class problems
        self.threshold = config.get('threshold', 0.95)
        self.w_cons = config.get('w_cons', 0.1)
        self.device = utils.get_device(config)
        self.bce = torch.nn.BCELoss(reduction=self.config.get("reduction", "mean"))

        # Bank to store samples from previous batches
        self.bank_features = []
        self.bank_probs = []

    def _update_bank(self, features, probabilities):
        """Update the bank with current batch samples"""
        self.bank_features = features.detach()
        self.bank_probs = probabilities.detach()

    def forward(self, feature_model, classifier_model, source_images, target_images, source_labels):
        feat_source = feature_model(source_images)
        
        output_source = classifier_model(feat_source, bypass_feature_extractor=True).to(float)
        prob_source = F.softmax(output_source, dim=1)
        prob_bar = F.softmax(output_source.detach_(), dim=1)

        output_target = classifier_model(target_images)
        prob_target = F.softmax(output_target, dim=1)

        feat_target = feature_model(target_images)

        # loss for adversarial adpative clustering
        aac_loss = self.unlabeled_loss(
            prob_source, prob_target, feat_target, source_labels, output_target
        )

        max_probs, pseudo_labels = torch.max(prob_target.detach_(), dim=-1)
        mask = max_probs.ge(self.threshold).float()
        # loss for pseudo labeling
        pl_loss = (F.cross_entropy(output_target, pseudo_labels, reduction='none') * mask).mean()

        prob_bar = F.softmax(output_source.detach_(), dim=1)
        prob_bar2 = F.softmax(output_target.detach_(), dim=1)
        # loss for consistency
        con_loss = self.w_cons * F.mse_loss(prob_bar, prob_bar2)

        loss = aac_loss + pl_loss + con_loss
        return loss

    def unlabeled_loss(self, prob_source, prob_target, feat_target, source_labels, output_target):
        # For labeled source data: compute similarity labels from class labels
        # Create similarity matrix from labels
        source_labels_row, source_labels_col = self.PairEnum(source_labels)
        similarity_labels = torch.sum(source_labels_row * source_labels_col, dim=-1)
        # Compute pairwise similarities between source samples
        prob_row, prob_col = self.PairEnum(prob_source)
        similarity_scores = torch.sum(prob_row * prob_col, dim=-1)  # ˆs_ij = p_i^T * p_j
        # Binary cross-entropy loss

        aac_loss = self.bce(similarity_scores.unsqueeze(1).double(), similarity_labels.unsqueeze(1).double())
        prob_target = F.softmax(output_target, dim=1)
        
        # Update bank with current target samples
        self._update_bank(feat_target, prob_target)
        
        # Compute AC loss between target and bank samples
        if len(self.bank_probs) > 0:
            
            # Compute similarity between current target and bank
            prob_target_row, bank_probs_col = self.PairEnum(prob_target)
            similarity_with_bank = torch.sum(prob_target_row * bank_probs_col, dim=-1)
            
            # Similarity labels from bank
            target_labels_row, bank_labels_col = self.PairEnum(torch.arange(prob_target.size(0)).to(self.device))
            similarity_labels_bank = (target_labels_row == bank_labels_col).float()
            
            aac_loss += self.bce(similarity_with_bank.unsqueeze(1), similarity_labels_bank.unsqueeze(1))

        return aac_loss

    def pairwise_target(self, feat, target):
        feat_detach = feat.detach()
        # For unlabeled data
        if target is None:
            rank_feat = feat_detach
            rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
            rank_idx1, rank_idx2 = self.PairEnum(rank_idx)
            rank_idx1, rank_idx2 = rank_idx1[:, :self.top_k], rank_idx2[:, :self.top_k]
            rank_idx1, _ = torch.sort(rank_idx1, dim=1)
            rank_idx2, _ = torch.sort(rank_idx2, dim=1)
            rank_diff = rank_idx1 - rank_idx2
            rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
            target_ulb = torch.ones_like(rank_diff).float().to(self.device)
            target_ulb[rank_diff > 0] = 0
        # For labeled data
        elif target is not None:
            target_row, target_col = self.PairEnum(target)
            target_ulb = torch.zeros(target.size(0) * target.size(0)).float().to(self.device)
            target_ulb[target_row == target_col] = 1
        return target_ulb


    def PairEnum(self, x):
        if x.ndimension() == 1:
            x1 = x.repeat(x.size(0), )
            x2 = x.repeat(x.size(0)).view(-1,x.size(0)).transpose(1, 0).reshape(-1)
        elif x.ndimension() == 2: 
            x1 = x.repeat(x.size(0), 1)
            x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
        else: 
            raise ValueError(f"Input must be 1D or 2D tensor, instead found {x.ndimension()}D tensor - {x.shape}")
        return x1, x2

class EntropySeparationLoss(Loss):

    """Handle arbitrary shifts between source and target domains
    reference: https://proceedings.neurips.cc/paper/2020/file/bb7946e7d85c81a9e69fee1cea4a087c-Paper.pdf

    - Code adapted from  https://github.com/VisionLearningGroup/DANCE/blob/main/utils/loss.py
    """
    def __init__(self, config):
        super().__init__(config)
        self.hinge_margin = config.get('hinge_margin', 0.0001)

    def _hinge_fn(self, x):
        return torch.clamp(x, min=self.hinge_margin)
    
    def _entropy_fn(self, p):
        return torch.sum(p * torch.log(p+1e-5), 1)

    def _seperation_loss(self, probabilities, labels):
      probabilities = torch.softmax(probabilities, dim=-1)[:, :labels.shape[-1]]
      return -torch.mean(
            self._hinge_fn(
                torch.abs(self._entropy_fn(probabilities - labels)), 
            )
        )

    def forward(self, probabilities, labels, boundry_p, boundry_m):
        loss = self._seperation_loss(probabilities, labels)
        if torch.abs(loss - boundry_p) > boundry_m:
            return -1* torch.abs(loss - boundry_p)
        else:
            return torch.tensor(0.0, device=probabilities.device, requires_grad=True)

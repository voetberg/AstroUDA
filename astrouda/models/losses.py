from abc import ABC, abstractmethod

from torch.nn import CrossEntropyLoss
import torch 


class Loss(ABC): 
    def __init__(self, config):
        super().__init__()
        self.loss_fn = None
        self.config = config

    @abstractmethod
    def forward(self, inputs, targets):
        raise NotImplementedError


class ObjectiveLoss(Loss): 
    """Objective loss such that L = L_CE + lambda*(L_AC + L_ES)"""
    def __init__(self, config): 
        super().__init__(config)
        self._lambda = config.get('lambda', 0.005)

        ce_weight = ""  # NS/(K*n_k), NS = images in source data, K = N source classes, n_k = images in class k
        self.loss_fn_ce = CrossEntropyLoss(weight=ce_weight, reduction='mean')
        self.loss_fn_ac = AdaptiveClusteringLoss(config)
        self.loss_fn_es = EntropySeparationLoss(config)

    def forward(self, probabilities, labels):

        ce_weight = ""  # NS/(K*n_k), NS = images in source data, K = N source classes, n_k = images in class k
        self.loss_fn_ce = CrossEntropyLoss(weight=ce_weight, reduction='mean')

        return (self.loss_fn_ce(inputs, targets) + 
                self._lambda * (self.loss_fn_ac(inputs, targets) + 
                                self.loss_fn_es(inputs, targets)))


class EntropyLoss(Loss):
    def __init__(self, config):
        super().__init__(config)

    def _compute_weight(self, labels):
        """
        Compute class weights based on the frequency of each class in the labels.
        """


    def forward(self, inputs, targets):
        weights = self._compute_weight(targets)
        loss_fn_ce = CrossEntropyLoss(weight=weights, reduction='mean')
        return loss_fn_ce(inputs, targets)


class AdaptiveClusteringLoss(Loss):
    '''Adaptive clustering loss via https://openaccess.thecvf.com/content/CVPR2021/supplemental/Li_Cross-Domain_Adaptive_Clustering_CVPR_2021_supplemental.pdf
    
    Reference: https://github.com/lijichang/CVPR2021-SSDA/blob/main/losses.py

    Produce a pairwise similarity of similar classes across domains
    '''
    def __init__(self, config):
        super().__init__(config)
        self.top_k = config.get('top_k', 7)  # Pick 3 for 3 class problems, 7 for 10 class problems

    def forward(self, probabilities, labels):
        ""

    def pairwise_similarity(self):
        """
        Compute pairwise similarity between unlabeled target samples x_1, x_2
        Uses the classier output probabilities p_1, p_2
        """

    def preprocess(self):
        """
        According to the paper - 90deg rotation & a zoom/crop versions are added to the dataset
        
        Given the same label as the original image. 
        """


class EntropySeparationLoss(Loss):

    """Handle arbitrary shifts between source and target domains
    reference: https://proceedings.neurips.cc/paper/2020/file/bb7946e7d85c81a9e69fee1cea4a087c-Paper.pdf

    - Code adapted from  https://github.com/VisionLearningGroup/DANCE/blob/main/utils/loss.py
    """
    def __init__(self, config):
        super().__init__(config)
        self.hinge_margin = config.get('hinge_margin', 0.2)

    def _hinge_fn(self, x):
        return torch.clamp(x, min=self.hinge_margin)
    
    def _entropy_fn(self, p):
        return torch.sum(p * torch.log(p+1e-5), 1)

    def forward(self, probabilities, labels):
        return -torch.mean(
            self._hinge_fn(
                torch.abs(-self._entropy_fn(probabilities) - labels), 
            )
        )

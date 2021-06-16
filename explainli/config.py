from enum import Enum
from typing import Optional


class AggregationMethods(Enum):
    L2 = 0
    SUM = 1
    MEAN = 2


class ForwardScoringOptions(Enum):
    TOP_PREDICTION = 0
    LOSS = 1
    PREDICTION_CLASS = 2


class AttributionMethods(Enum):
    InputXGradient = 0
    DeepLift = 1
    GradientShap = 2
    LayerConductance = 3
    Activation = 4
    IntegratedGradients = 5
    GuidedBackprop = 6
    Saliency = 7
    Shapley = 8
    Occlusion = 9


class AttributionConfig(object):
    def __init__(self, attribution_method: AttributionMethods, remove_pad_tokens: bool, remove_sep_tokens: bool,
                 remove_cls_token: bool, join_subwords: bool,
                 aggregation_method: AggregationMethods, forward_scoring: ForwardScoringOptions, normalize_scores: bool,
                 prediction_class_idx: Optional[int]=None):
        """
        Common configuration for all the attribution methods supported
        :param attribution_method:
        :param remove_pad_tokens:
        :param remove_sep_tokens:
        :param remove_cls_token:
        :param join_subwords:
        :param aggregation_method: aggregation method used for aggregation feature attributions to create
        token score (L2, mean, sum)
        :param forward_scoring: the value to be returned by model's forward to calculate attributions wrt loss,
         top predicted class or any arbitrary class
        :param normalize_scores:
        :param prediction_class_idx: if forward scoring selected as PREDICTION_CLASS, prediction class index must be given
        """
        self.attribution_method = attribution_method
        self.remove_pad_tokens = remove_pad_tokens
        self.remove_cls_token = remove_cls_token
        self.remove_sep_tokens = remove_sep_tokens
        self.join_subwords = join_subwords
        self.aggregation_method = aggregation_method
        self.forward_scoring = forward_scoring
        self.normalize_scores = normalize_scores
        self.prediction_class_idx = prediction_class_idx

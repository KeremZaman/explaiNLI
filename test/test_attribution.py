import unittest
from explainli.config import AttributionMethods, AttributionConfig, AggregationMethods, ForwardScoringOptions
from explainli.explainli import NLIAttribution
from captum._utils.models.linear_model import SkLearnLasso
from captum.attr._core.lime import get_exp_kernel_similarity_function
import numpy as np
import torch


class AttributionTestBase(unittest.TestCase):
    def __init__(self, method, scoring, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method
        self.scoring = scoring
        self.model_name = 'textattack/bert-base-uncased-snli'
        self.kwargs = kwargs

        self.attr_config = AttributionConfig(self.method, remove_pad_tokens=True,
                                             remove_cls_token=False, remove_sep_tokens=False, join_subwords=False,
                                             normalize_scores=True,
                                             forward_scoring=self.scoring,
                                             aggregation_method=AggregationMethods.MEAN)

    def setUp(self):
        self.attribution = NLIAttribution(model_name=self.model_name, config=self.attr_config, **self.kwargs)

    def _test_batch_versus_single_scores(self, **kwargs):
        """
        Test whether attribution calculations are same when they are calculated together inside a batch and separately
         with consecutive calls
        :param kwargs:
        :return:
        """
        inputs = [("A man inspects the uniform of a figure in some East Asian country.", "The man is sleeping"),
                  ("A soccer game with multiple males playing.", "Some men are playing a sport."),
                  ("A black race car starts up in front of a crowd of people.", "A man is driving down a lonely road."),
                  ("A smiling costumed woman is holding an umbrella.", "A happy woman in a fairy costume holds an umbrella.")]
        labels = [2, 0, 2, 1]

        attribution_single = NLIAttribution(model_name=self.model_name, config=self.attr_config)

        for pair, label in zip(inputs, labels):
            attribution_single.attr([pair], [label], **kwargs)

        self.attribution.attr(inputs, labels, **kwargs)
        for single_record, batch_record in zip(attribution_single.records, self.attribution.records):
            single_scores = single_record.word_attributions
            batch_scores = batch_record.word_attributions

            self.assertTrue(np.allclose(single_scores, batch_scores, atol=1e-5))

    def _test_consistency_inside_batch(self, **kwargs):
        """
        Test whether methods give same scores to same examples inside the same batch
        :param kwargs:
        :return:
        """
        inputs = [("A man inspects the uniform of a figure in some East Asian country.", "The man is sleeping"),
                  ("A soccer game with multiple males playing.", "Some men are playing a sport."),
                  ("A black race car starts up in front of a crowd of people.", "A man is driving down a lonely road."),
                  ("A smiling costumed woman is holding an umbrella.", "A happy woman in a fairy costume holds an umbrella."),
                  ("A soccer game with multiple males playing.", "Some men are playing a sport."),
                  ("A man inspects the uniform of a figure in some East Asian country.", "The man is sleeping"),
                  ("A smiling costumed woman is holding an umbrella.", "A happy woman in a fairy costume holds an umbrella."),
                  ("A black race car starts up in front of a crowd of people.", "A man is driving down a lonely road.")]
        labels = [2, 0, 2, 1, 0, 2, 1, 2]

        self.attribution.attr(inputs, labels, **kwargs)

        scores = np.zeros((len(self.attribution.records), max([len(record.word_attributions) for record in self.attribution.records])), dtype=np.float)
        for i, record in enumerate(self.attribution.records):
            scores[i, :len(record.word_attributions)] = record.word_attributions

        self.assertTrue(np.allclose(scores[0, :], scores[5, :], atol=1e-5))
        self.assertTrue(np.allclose(scores[1, :], scores[4, :], atol=1e-5))
        self.assertTrue(np.allclose(scores[2, :], scores[7, :], atol=1e-5))
        self.assertTrue(np.allclose(scores[3, :], scores[6, :], atol=1e-5))

    def _test_consistency_across_time(self, **kwargs):
        """
        Test whether methods give the same scores after a number of repetitions. This test mainly aims to check gradient
        accumulation.
        :param kwargs:
        :return:
        """
        inputs = [("A man inspects the uniform of a figure in some East Asian country.", "The man is sleeping"),
                  ("A soccer game with multiple males playing.", "Some men are playing a sport."),
                  ("A black race car starts up in front of a crowd of people.", "A man is driving down a lonely road."),
                  ("A smiling costumed woman is holding an umbrella.", "A happy woman in a fairy costume holds an umbrella.")]
        labels = [2, 0, 2, 1]
        n = len(inputs)

        scores_step_01, scores_step_05, scores_step_10, scores_step_25 = None, None, None, None

        for i in range(1, 26):
            torch.manual_seed(0)
            self.attribution.attr(inputs, labels, **kwargs)

            scores = np.zeros((n,
                               max([len(record.word_attributions) for record in self.attribution.records[-n:]])),
                              dtype=np.float)
            for j, record in enumerate(self.attribution.records[-n:]):
                scores[j, :len(record.word_attributions)] = record.word_attributions[:]

            if i == 1:
                scores_step_01 = scores[:]
            elif i == 5:
                scores_step_05 = scores[:]
            elif i == 10:
                scores_step_10 = scores[:]
            elif i == 25:
                scores_step_25 = scores[:]

        self.assertTrue(np.allclose(scores_step_01, scores_step_05))
        self.assertTrue(np.allclose(scores_step_05, scores_step_10))
        self.assertTrue(np.allclose(scores_step_10, scores_step_25))


class InputXGradientWrtTopPredictionAttributionTest(AttributionTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(AttributionMethods.InputXGradient, ForwardScoringOptions.TOP_PREDICTION, *args, **kwargs)

    def test_batch_versus_single_scores(self):
        self._test_batch_versus_single_scores()

    def test_consistency_inside_batch(self):
        self._test_consistency_inside_batch()

    def test_consistency_across_time(self):
        self._test_consistency_across_time()


class InputXGradientWrtLossAttributionTest(AttributionTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(AttributionMethods.InputXGradient, ForwardScoringOptions.LOSS, *args, **kwargs)

    def test_batch_versus_single_scores(self):
        self._test_batch_versus_single_scores()

    def test_consistency_inside_batch(self):
        self._test_consistency_inside_batch()

    def test_consistency_across_time(self):
        self._test_consistency_across_time()


class SaliencyWrtTopPredictionAttributionTest(AttributionTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(AttributionMethods.Saliency, ForwardScoringOptions.TOP_PREDICTION, *args, **kwargs)

    def test_batch_versus_single_scores(self):
        self._test_batch_versus_single_scores()

    def test_consistency_inside_batch(self):
        self._test_consistency_inside_batch()

    def test_consistency_across_time(self):
        self._test_consistency_across_time()


class SaliencyWrtLossAttributionTest(AttributionTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(AttributionMethods.Saliency, ForwardScoringOptions.LOSS, *args, **kwargs)

    def test_batch_versus_single_scores(self):
        self._test_batch_versus_single_scores()

    def test_consistency_inside_batch(self):
        self._test_consistency_inside_batch()

    def test_consistency_across_time(self):
        self._test_consistency_across_time()


class ActivationWrtTopPredictionAttributionTest(AttributionTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(AttributionMethods.Activation, ForwardScoringOptions.TOP_PREDICTION, *args, **kwargs)

    def test_batch_versus_single_scores(self):
        self._test_batch_versus_single_scores()

    def test_consistency_inside_batch(self):
        self._test_consistency_inside_batch()

    def test_consistency_across_time(self):
        self._test_consistency_across_time()


class ActivationWrtLossAttributionTest(AttributionTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(AttributionMethods.Activation, ForwardScoringOptions.LOSS, *args, **kwargs)

    def test_batch_versus_single_scores(self):
        self._test_batch_versus_single_scores()

    def test_consistency_inside_batch(self):
        self._test_consistency_inside_batch()

    def test_consistency_across_time(self):
        self._test_consistency_across_time()


class DeepLiftWrtTopPredictionAttributionTest(AttributionTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(AttributionMethods.DeepLift, ForwardScoringOptions.TOP_PREDICTION, *args, **kwargs)

    def test_batch_versus_single_scores(self):
        self._test_batch_versus_single_scores()

    def test_consistency_inside_batch(self):
        self._test_consistency_inside_batch()

    def test_consistency_across_time(self):
        self._test_consistency_across_time()


class DeepLiftWrtLossAttributionTest(AttributionTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(AttributionMethods.DeepLift, ForwardScoringOptions.LOSS, *args, **kwargs)

    def test_batch_versus_single_scores(self):
        self._test_batch_versus_single_scores()

    def test_consistency_inside_batch(self):
        self._test_consistency_inside_batch()

    def test_consistency_across_time(self):
        self._test_consistency_across_time()


class GuidedBackpropWrtTopPredictionAttributionTest(AttributionTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(AttributionMethods.GuidedBackprop, ForwardScoringOptions.TOP_PREDICTION, *args, **kwargs)

    def test_batch_versus_single_scores(self):
        self._test_batch_versus_single_scores()

    def test_consistency_inside_batch(self):
        self._test_consistency_inside_batch()

    def test_consistency_across_time(self):
        self._test_consistency_across_time()


class GuidedBackpropWrtLossAttributionTest(AttributionTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(AttributionMethods.GuidedBackprop, ForwardScoringOptions.LOSS, *args, **kwargs)

    def test_batch_versus_single_scores(self):
        self._test_batch_versus_single_scores()

    def test_consistency_inside_batch(self):
        self._test_consistency_inside_batch()

    def test_consistency_across_time(self):
        self._test_consistency_across_time()


class OcclusionWrtTopPredictionAttributionTest(AttributionTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(AttributionMethods.Occlusion, ForwardScoringOptions.TOP_PREDICTION, *args, **kwargs)

    def test_batch_versus_single_scores(self):
        self._test_batch_versus_single_scores(sliding_window_shapes=(1, 768))

    def test_consistency_inside_batch(self):
        self._test_consistency_inside_batch(sliding_window_shapes=(1, 768))

    def test_consistency_across_time(self):
        self._test_consistency_across_time(sliding_window_shapes=(1, 768))


class OcclusionWrtLossAttributionTest(AttributionTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(AttributionMethods.Occlusion, ForwardScoringOptions.LOSS, *args, **kwargs)

    def test_batch_versus_single_scores(self):
        self._test_batch_versus_single_scores(sliding_window_shapes=(1, 768))

    def test_consistency_inside_batch(self):
        self._test_consistency_inside_batch(sliding_window_shapes=(1, 768))

    def test_consistency_across_time(self):
        self._test_consistency_across_time(sliding_window_shapes=(1, 768))


class ShapleyWrtTopPredictionAttributionTest(AttributionTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(AttributionMethods.Shapley, ForwardScoringOptions.TOP_PREDICTION, *args, **kwargs)

    def test_consistency_across_time(self):
        feature_mask = torch.range(0, 25).unsqueeze(1).repeat(1, 1, 768).to(self.attribution.device)
        self._test_consistency_across_time(n_samples=10, feature_mask=feature_mask)


class ShapleyWrtLossAttributionTest(AttributionTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(AttributionMethods.Shapley, ForwardScoringOptions.LOSS, *args, **kwargs)

    def test_consistency_across_time(self):
        feature_mask = torch.range(0, 25).unsqueeze(1).repeat(1, 1, 768).to(self.attribution.device)
        self._test_consistency_across_time(n_samples=10, feature_mask=feature_mask)


class IGWrtTopPredictionAttributionTest(AttributionTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(AttributionMethods.IntegratedGradients, ForwardScoringOptions.TOP_PREDICTION, *args, **kwargs)

    def test_batch_versus_single_scores(self):
        self._test_batch_versus_single_scores(n_steps=25)

    def test_consistency_inside_batch(self):
        self._test_consistency_inside_batch(n_steps=25)

    def test_consistency_across_time(self):
        self._test_consistency_across_time(n_steps=25)


class IGWrtLossAttributionTest(AttributionTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(AttributionMethods.IntegratedGradients, ForwardScoringOptions.LOSS, *args, **kwargs)

    def test_batch_versus_single_scores(self):
        self._test_batch_versus_single_scores(n_steps=25)

    def test_consistency_inside_batch(self):
        self._test_consistency_inside_batch(n_steps=25)

    def test_consistency_across_time(self):
        self._test_consistency_across_time(n_steps=25)


class LIMEWrtTopPredictionAttributionTest(AttributionTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(AttributionMethods.LIME, ForwardScoringOptions.TOP_PREDICTION, *args, **kwargs)

    def setUp(self):
        self.attribution = NLIAttribution(model_name=self.model_name, config=self.attr_config,
                                          interpretable_model=SkLearnLasso(alpha=0.08),
                                          similarity_func=get_exp_kernel_similarity_function('cosine', 1.0),
                                          **self.kwargs)

    def test_consistency_across_time(self):
        feature_mask = torch.range(0, 25).unsqueeze(1).repeat(1, 1, 768).to(self.attribution.device).long()
        self._test_consistency_across_time(n_samples=20, feature_mask=feature_mask, perturbations_per_eval=8)


class LIMEWrtLossAttributionTest(AttributionTestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(AttributionMethods.LIME, ForwardScoringOptions.LOSS, *args, **kwargs)

    def setUp(self):
        self.attribution = NLIAttribution(model_name=self.model_name, config=self.attr_config,
                                          interpretable_model=SkLearnLasso(alpha=0.08),
                                          similarity_func=get_exp_kernel_similarity_function('cosine', 1.0),
                                          **self.kwargs)

    def test_consistency_across_time(self):
        feature_mask = torch.range(0, 25).unsqueeze(1).repeat(1, 1, 768).to(self.attribution.device).long()
        self._test_consistency_across_time(n_samples=20, feature_mask=feature_mask, perturbations_per_eval=1)

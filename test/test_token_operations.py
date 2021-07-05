import unittest
from explainli.config import AttributionMethods, AttributionConfig, AggregationMethods, ForwardScoringOptions
from explainli.explainli import NLIAttribution


class BaseTest(unittest.TestCase):
    def __init__(self, method):
        super().__init__()
        self.method = method
        self.inputs = [("A man inspects the uniform of a figure in some East Asian country.", "The man is sleeping"),
                       ("A man is playing basketball now.", "The man is kicking the ball.")]
        self.labels = [2, 0]
        self.model_name = 'textattack/bert-base-uncased-snli'


class TokenRemovalBaseTest(BaseTest):
    def __init__(self, method):
        super().__init__(method)

    def setUp(self):
        self.attr_config = AttributionConfig(self.method, remove_pad_tokens=False,
                                             remove_cls_token=False, remove_sep_tokens=True, join_subwords=False,
                                             normalize_scores=False,
                                             forward_scoring=ForwardScoringOptions.TOP_PREDICTION,
                                             aggregation_method=AggregationMethods.MEAN)
        self.attribution = NLIAttribution(model_name=self.model_name, config=self.attr_config)

    def _test_remove_token(self):
        self.attribution.attr(self.inputs, self.labels)

        for record in self.attribution.records:
            scores = record.attr_score
            input = record.raw_input

            # check if input contains removed token
            self.assertNotIn(self.token, input)

            # check if scores for the removed token are deleted
            self.assertEqual(len(scores), len(input))


class SepTokenRemovalBaseTest(TokenRemovalBaseTest):
    def __init__(self, method):
        super().__init__(method)

    def setUp(self):
        super().setUp()
        self.attr_config = AttributionConfig(self.method, remove_pad_tokens=False,
                                             remove_cls_token=False, remove_sep_tokens=True, join_subwords=False,
                                             normalize_scores=False,
                                             forward_scoring=ForwardScoringOptions.TOP_PREDICTION,
                                             aggregation_method=AggregationMethods.MEAN)
        self.attribution = NLIAttribution(model_name=self.model_name, config=self.attr_config)

    def _test_remove_sep_token(self):
        self._test_remove_token()


class PadTokenRemovalBaseTest(TokenRemovalBaseTest):
    def __init__(self, method):
        super().__init__(method)

    def setUp(self):
        super().setUp()
        self.attr_config = AttributionConfig(self.method, remove_pad_tokens=True,
                                             remove_cls_token=False, remove_sep_tokens=False, join_subwords=False,
                                             normalize_scores=False,
                                             forward_scoring=ForwardScoringOptions.TOP_PREDICTION,
                                             aggregation_method=AggregationMethods.MEAN)
        self.attribution = NLIAttribution(model_name=self.model_name, config=self.attr_config)

    def _test_remove_pad_token(self):
        self._test_remove_token()


class ClsTokenRemovalBaseTest(TokenRemovalBaseTest):
    def __init__(self, method):
        super().__init__(method)

    def setUp(self):
        super().setUp()
        self.attr_config = AttributionConfig(self.method, remove_pad_tokens=False,
                                             remove_cls_token=True, remove_sep_tokens=False, join_subwords=False,
                                             normalize_scores=False,
                                             forward_scoring=ForwardScoringOptions.TOP_PREDICTION,
                                             aggregation_method=AggregationMethods.MEAN)
        self.attribution = NLIAttribution(model_name=self.model_name, config=self.attr_config)

    def _test_remove_cls_token(self):
        self._test_remove_token()


class SubwordBaseTest(BaseTest):
    def __init__(self, method):
        super().__init__(method)

    def setUp(self):
        self.attr_config = AttributionConfig(self.method, remove_pad_tokens=False,
                                             remove_cls_token=False, remove_sep_tokens=False, join_subwords=True,
                                             normalize_scores=False,
                                             forward_scoring=ForwardScoringOptions.TOP_PREDICTION,
                                             aggregation_method=AggregationMethods.MEAN)
        self.attribution = NLIAttribution(model_name=self.model_name, config=self.attr_config)

    def _test_join_subwords(self):
        self.attribution.attr(self.inputs, self.labels)

        for record in self.attribution.records:
            scores = record.attr_score
            input = record.raw_input

            # check if scores for the joined tokens are merged
            self.assertEqual(len(scores), len(input))

            for token in input:
                self.assertIs(token.startswith('##'), False)


class InputXGradientSepTokenRemovalTest(SepTokenRemovalBaseTest):
    def __init__(self):
        super().__init__(AttributionMethods.InputXGradient)

    def test_remove_sep_token(self):
        self._test_remove_sep_token()


class InputXGradientPadTokenRemovalTest(PadTokenRemovalBaseTest):
    def __init__(self):
        super().__init__(AttributionMethods.InputXGradient)

    def test_remove_pad_token(self):
        self._test_remove_pad_token()


class InputXGradientClsTokenRemovalTest(ClsTokenRemovalBaseTest):
    def __init__(self):
        super().__init__(AttributionMethods.InputXGradient)

    def test_remove_pad_token(self):
        self._test_remove_cls_token()


class InputXGradientSubwordTest(SubwordBaseTest):
    def __init__(self):
        super().__init__(AttributionMethods.InputXGradient)

    def _test_join_subwords(self):
        self._test_join_subwords()

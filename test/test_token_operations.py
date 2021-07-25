import unittest
from explainli.config import AttributionMethods, AttributionConfig, AggregationMethods, ForwardScoringOptions
from explainli.explainli import NLIAttribution


class BaseTest(unittest.TestCase):
    def __init__(self, method, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method
        self.inputs = [("A man inspects the uniform of a figure in some East Asian country.", "The man is sleeping"),
                       ("A man is playing basketball now.", "The man is kicking the ball."),
                       ('"A woman with a green headscarf, blue shirt and a very big grin."', 'The woman is young.')]
        self.labels = [2, 0, 1]
        self.model_name = 'textattack/bert-base-uncased-snli'


class TokenRemovalBaseTest(BaseTest):
    def __init__(self, method, *args, **kwargs):
        super().__init__(method, *args, **kwargs)

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
            scores = record.word_attributions
            input = record.raw_input

            # check if input contains removed token
            self.assertNotIn(self.token, input)

            # check if scores for the removed token are deleted
            self.assertEqual(len(scores), len(input))


class SepTokenRemovalBaseTest(TokenRemovalBaseTest):
    def __init__(self, method, *args, **kwargs):
        super().__init__(method, *args, **kwargs)

    def setUp(self):
        super().setUp()
        self.attr_config = AttributionConfig(self.method, remove_pad_tokens=False,
                                             remove_cls_token=False, remove_sep_tokens=True, join_subwords=False,
                                             normalize_scores=False,
                                             forward_scoring=ForwardScoringOptions.TOP_PREDICTION,
                                             aggregation_method=AggregationMethods.MEAN)
        self.attribution = NLIAttribution(model_name=self.model_name, config=self.attr_config)
        self.token = self.attribution.tokenizer.sep_token

    def _test_remove_sep_token(self):
        self._test_remove_token()


class PadTokenRemovalBaseTest(TokenRemovalBaseTest):
    def __init__(self, method, *args, **kwargs):
        super().__init__(method, *args, **kwargs)

    def setUp(self):
        super().setUp()
        self.attr_config = AttributionConfig(self.method, remove_pad_tokens=True,
                                             remove_cls_token=False, remove_sep_tokens=False, join_subwords=False,
                                             normalize_scores=False,
                                             forward_scoring=ForwardScoringOptions.TOP_PREDICTION,
                                             aggregation_method=AggregationMethods.MEAN)
        self.attribution = NLIAttribution(model_name=self.model_name, config=self.attr_config)
        self.token = self.attribution.tokenizer.pad_token

    def _test_remove_pad_token(self):
        self._test_remove_token()


class ClsTokenRemovalBaseTest(TokenRemovalBaseTest):
    def __init__(self, method, *args, **kwargs):
        super().__init__(method, *args, **kwargs)

    def setUp(self):
        super().setUp()
        self.attr_config = AttributionConfig(self.method, remove_pad_tokens=False,
                                             remove_cls_token=True, remove_sep_tokens=False, join_subwords=False,
                                             normalize_scores=False,
                                             forward_scoring=ForwardScoringOptions.TOP_PREDICTION,
                                             aggregation_method=AggregationMethods.MEAN)
        self.attribution = NLIAttribution(model_name=self.model_name, config=self.attr_config)
        self.token = self.attribution.tokenizer.cls_token

    def _test_remove_cls_token(self):
        self._test_remove_token()


class SubwordBaseTest(BaseTest):
    def __init__(self, method, *args, **kwargs):
        super().__init__(method, *args, **kwargs)

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
            scores = record.word_attributions
            input = record.raw_input

            # check if scores for the joined tokens are merged
            self.assertEqual(len(scores), len(input))

            for token in input:
                self.assertIs(token.startswith('##'), False)


class InputXGradientSepTokenRemovalTest(SepTokenRemovalBaseTest):
    def __init__(self, *args, **kwargs):
        super().__init__(AttributionMethods.InputXGradient, *args, **kwargs)

    def test_remove_sep_token(self):
        self._test_remove_sep_token()


class InputXGradientPadTokenRemovalTest(PadTokenRemovalBaseTest):
    def __init__(self, *args, **kwargs):
        super().__init__(AttributionMethods.InputXGradient, *args, **kwargs)

    def test_remove_pad_token(self):
        self._test_remove_pad_token()


class InputXGradientClsTokenRemovalTest(ClsTokenRemovalBaseTest):
    def __init__(self,  *args, **kwargs):
        super().__init__(AttributionMethods.InputXGradient, *args, **kwargs)

    def test_remove_pad_token(self):
        self._test_remove_cls_token()


class InputXGradientSubwordTest(SubwordBaseTest):
    def __init__(self, *args, **kwargs):
        super().__init__(AttributionMethods.InputXGradient, *args, **kwargs)

    def _test_join_subwords(self):
        self._test_join_subwords()

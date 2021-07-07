import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from captum.attr import visualization as viz
from captum.attr import LayerIntegratedGradients, LayerGradientXActivation, LayerDeepLift, LayerActivation
from captum.attr import GuidedBackprop, Saliency, GradientShap, Occlusion, ShapleyValueSampling

from .config import AttributionConfig, AggregationMethods, ForwardScoringOptions, AttributionMethods
from functools import partial

from typing import Tuple, List, Optional


ATTRIBUTION_METHOD_MAP = {
    AttributionMethods.InputXGradient: LayerGradientXActivation,
    AttributionMethods.DeepLift: LayerDeepLift,
    AttributionMethods.Activation: LayerActivation,
    AttributionMethods.IntegratedGradients: LayerIntegratedGradients,
    AttributionMethods.GuidedBackprop: GuidedBackprop,
    AttributionMethods.Saliency: Saliency,
    AttributionMethods.Shapley: ShapleyValueSampling,
    AttributionMethods.Occlusion: Occlusion,
    AttributionMethods.GradientShap: GradientShap
}


class NLIAttribution(object):
    """
    A common class to apply a wide variety of attribution methods
    """

    def __init__(self, model_name: str, config: AttributionConfig, **kwargs):
        """

        :param model_name: NLI model to load from HuggingFace model hub
        :param config: Config object for common settings
        :param kwargs: Additional arguments that are specific to attribution method
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.wrapper = self.ModelWrapper(model_name, config, self.device)

        self.wrapper.to(self.device)
        self.wrapper.eval()

        self.records = []  # save attribution results

        self.aggregation_funcs = {
            AggregationMethods.L2: lambda x: torch.norm(x, 2, -1),
            AggregationMethods.SUM: lambda x: x.sum(dim=-1),
            AggregationMethods.MEAN: lambda x: x.mean(dim=-1)
        }

        self.aggregation_func = self.aggregation_funcs[config.aggregation_method]

        self.config = config
        self.label_names = self.config.label_names

        if self._is_input_attribution():
            self.attr_method = ATTRIBUTION_METHOD_MAP.get(self.config.attribution_method)(self.wrapper, **kwargs)
        else:
            self.attr_method = ATTRIBUTION_METHOD_MAP.get(self.config.attribution_method)(self.wrapper,
                                                                                          layer=self.wrapper.model.bert.embeddings,
                                                                                          **kwargs)

    def _is_baseline_required(self) -> bool:
        """
        Since some attribution methods require baselines, we need to create baseline input for them

        :return:
        """
        if self.config.attribution_method in [AttributionMethods.IntegratedGradients, AttributionMethods.GradientShap]:
            return True
        else:
            return False

    def _is_input_attribution(self) -> bool:
        """
        Some of the methods is implemented by using Layer Attribution of captum for convenience. The other ones, that
        use Input Attribution, require the input embeddings of BERT model to be given as input instead of token ids.

        :return:
        """
        if self.config.attribution_method in [AttributionMethods.GuidedBackprop, AttributionMethods.Saliency,
                                              AttributionMethods.Shapley, AttributionMethods.GradientShap,
                                              AttributionMethods.Occlusion]:
            return True
        else:
            return False

    def _construct_baseline_input(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor,
                                                                          torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create baseline inputs for the methods requiring baseline
        Baseline inputs are all zero vectors for each token in input except SEP and CLS token (like at the captum
        tutorials)

        :param input_ids:
        :return:
        """
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id

        ref_input_ids = torch.where((input_ids == cls_token_id) | (input_ids == sep_token_id), input_ids, 0)
        ref_attn_mask = torch.ones_like(input_ids)
        ref_token_type_ids = torch.zeros_like(input_ids)
        ref_position_ids = torch.zeros_like(input_ids)

        return ref_input_ids, ref_attn_mask, ref_token_type_ids, ref_position_ids

    def _nli_forward_plain(self, input_ids: torch.Tensor, labels: List[int], token_type_ids: Optional[torch.Tensor]=None,
                           position_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None,
                           output_hidden_states: bool=False):
        """
        Return plain predictions of model to be used for intermediate steps such as getting input embeddings

        :param input_ids:
        :param labels:
        :param token_type_ids:
        :param position_ids:
        :param attention_mask:
        :param output_hidden_states:
        :return:
        """
        labels = torch.LongTensor(labels, device=self.device)
        self.wrapper.zero_grad()
        pred = self.wrapper.model(input_ids, token_type_ids=token_type_ids, position_ids=position_ids,
                                  attention_mask=attention_mask, labels=labels,
                                  output_hidden_states=output_hidden_states)

        return pred

    def attr(self, inputs: List[Tuple[str, str]], labels: List[int], return_abs: Optional[bool]=False, **kwargs) -> torch.Tensor:
        """
        Calculate attributions, postprocess, save and return raw attributions
        :param inputs:
        :param labels:
        :param return_abs: if set True, return absolute value of aggregated attributions
        :param kwargs:
        :return:
        """
        encoded_inputs = self.tokenizer(inputs, return_tensors='pt', padding=True)

        # construct baseline input if the selected attribution method requires
        if self._is_baseline_required():
            ref_input_ids, ref_attn_mask, ref_token_type_ids, ref_position_ids = self._construct_baseline_input(
                encoded_inputs['input_ids'])

            # hide baseline input into attribute function to be able use it in a generalized way
            self.attr_method.attribute = partial(self.attr_method.attribute, baselines=ref_input_ids)

        # if select method is implemented as input attribution, make model use input embeddings as input
        if self._is_input_attribution():
            # first element of hidden states is input embeddings
            inputs_embeds = self._nli_forward_plain(input_ids=encoded_inputs['input_ids'], labels=labels,
                                                    token_type_ids=encoded_inputs['token_type_ids'],
                                                    attention_mask=encoded_inputs['attention_mask'],
                                                    output_hidden_states=True).hidden_states[0]
            # set use_embeds_as_inputs True
            additional_args = (labels, encoded_inputs['token_type_ids'], None, encoded_inputs['attention_mask'], True)
            inputs = inputs_embeds
        else:
            additional_args = (labels, encoded_inputs['token_type_ids'], None, encoded_inputs['attention_mask'], False)
            inputs = encoded_inputs['input_ids']

        self.wrapper.zero_grad()
        attributions = self.attr_method.attribute(inputs, additional_forward_args=additional_args, **kwargs)
        attributions = self.aggregation_func(attributions)
        attributions = torch.abs(attributions) if return_abs else attributions

        indices_list = [input_ids.detach().tolist() for input_ids in encoded_inputs['input_ids']]
        tokens_list = [self.tokenizer.convert_ids_to_tokens(indices) for indices in indices_list]
        attribution_list = attributions.detach().tolist()

        # get predictions separately to get prediction scores and predicted labels
        pred = self._nli_forward_plain(encoded_inputs['input_ids'], labels, encoded_inputs['token_type_ids'], None,
                                       encoded_inputs['attention_mask'])
        pred_list = torch.softmax(pred['logits'], dim=-1).detach().tolist()[-len(labels):]

        for tokens, attribution_item, pred, label in zip(tokens_list, attribution_list, pred_list, labels):

            # if it's set True, remove padding tokens from tokens and their scores from attribution scores
            # it's expected that they have zero attribution
            if self.config.remove_pad_tokens and self.tokenizer.pad_token in tokens:
                padding_start = tokens.index(self.tokenizer.pad_token)
                tokens = tokens[:padding_start]
                attribution_item = attribution_item[:padding_start]

            # if it's set True remove SEP tokens from tokens and their scores from attribution scores
            # it can be preferred not only for visual purposes but also for excluding effect of SEP tokens to evaluate
            # properly
            if self.config.remove_sep_tokens:
                while self.tokenizer.sep_token in tokens:
                    sep_idx = tokens.index(self.tokenizer.sep_token)
                    tokens.pop(sep_idx)
                    attribution_item.pop(sep_idx)

            # if it's set True remove CLS tokens from tokens and their scores from attribution scores
            # it can be preferred for the same reasons as SEP token
            if self.config.remove_cls_token and self.tokenizer.cls_token in tokens:
                tokens.pop(0)
                attribution_item.pop(0)

            # attributions are calculated for each token. it may be needed to get word-wise scores for visualization and
            # evaluation purposes
            if self.config.join_subwords:
                for i, token in enumerate(tokens):
                    if token.startswith('##'):
                        token = token.replace('##', '')
                        tokens[i - 1] = tokens[i - 1] + token
                        tokens.pop(i)
                        attribution_item[i - 1] += attribution_item[i]
                        attribution_item.pop(i)

            # normalize scores between (-1, 1)
            if self.config.normalize_scores:
                attribution_item = (torch.Tensor(attribution_item) / torch.norm(
                    torch.Tensor(attribution_item))).detach().tolist()

            # create visualization instance
            record = viz.VisualizationDataRecord(
                attribution_item,
                max(pred),
                self.label_names[pred.index(max(pred))],
                self.label_names[label],
                str(self.label_names[label]),
                sum(attribution_item),
                tokens, 0.0)

            self.records.append(record)

        return attributions

    def visualize(self):
        """
        Visualize saved attributions
        :return:
        """
        viz.visualize_text(self.records)

    class ModelWrapper(nn.Module):
        """
        To calculate attributions wrt loss, top predicted class or any arbitrary class, forward method needs to return
        different values. It might be implemented in an additional forward_func in main class but some captum methods
        just accept model's forward. So this is a wrapper for conditional forward method
        """

        def __init__(self, model_name, config, device):
            super().__init__()
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.device = device
            self.config = config

        def forward(self, inputs, labels=None, token_type_ids=None, position_ids=None, attention_mask=None,
                    use_embeds_as_input=False):
            labels = torch.LongTensor(labels, device=self.device)

            # in case of batching multiple steps (e.g integrated gradients)
            if labels.shape != inputs.shape:
                labels = labels.repeat(inputs.shape[0] // labels.shape[0], 1)

            if use_embeds_as_input:
                pred = self.model(inputs_embeds=inputs, token_type_ids=token_type_ids,
                                  position_ids=position_ids, attention_mask=attention_mask, labels=labels)

            else:
                pred = self.model(inputs, token_type_ids=token_type_ids,
                                  position_ids=position_ids, attention_mask=attention_mask, labels=labels)

            if self.config.forward_scoring is ForwardScoringOptions.LOSS:
                return pred['loss'].unsqueeze(0)
            elif self.config.forward_scoring is ForwardScoringOptions.TOP_PREDICTION:
                return pred['logits'].max(1).values
            elif self.config.forward_scoring is ForwardScoringOptions.PREDICTION_CLASS:
                return pred['logits'][:, self.config.prediction_class_idx]

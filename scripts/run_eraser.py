from explainli.config import AttributionMethods, AttributionConfig, AggregationMethods, ForwardScoringOptions
from explainli.explainli import NLIAttribution
from eval import eraser_faithfulness

# from IPython.display import display
import pandas as pd
import numpy as np

import json
import argparse
import os
import torch


def save_results(attribution: NLIAttribution, nli_model_name: str, comprehensiveness_score: float,
                 sufficiency_score: float, output_dir: str) -> None:
    """
    Save ERASER scores (comprehensiveness and sufficiency)

    :param attribution:
    :param nli_model_name:
    :param comprehensiveness_score:
    :param sufficiency_score:
    :param output_dir:
    """
    attr_method = getattr(attribution.config, 'attribution_method').name
    agg_method = getattr(attribution.config, 'aggregation_method').name.lower()
    output_mechanism = getattr(attribution.config, 'forward_scoring').name.lower()
    model_name = nli_model_name.split('/')[-1]

    data = {
        'comprehensiveness': comprehensiveness_score.item(),
        'sufficiency_score': sufficiency_score.item()
    }

    with open(f'{output_dir}/{attr_method}_{agg_method}_wrt_{output_mechanism}_{model_name}.json', 'w') as log_file:
        json.dump(data, log_file, indent=6)


if __name__ == '__main__':

    aggregation_methods = {
        'mean': AggregationMethods.MEAN,
        'sum': AggregationMethods.SUM,
        'l2': AggregationMethods.L2
    }

    output_methods = {
        'tp': ForwardScoringOptions.TOP_PREDICTION,
        'loss': ForwardScoringOptions.LOSS
    }

    attr_methods = {
        'ig': AttributionMethods.IntegratedGradients,
        'inputxgrad': AttributionMethods.InputXGradient,
        'saliency': AttributionMethods.Saliency,
        'activation': AttributionMethods.Activation,
        'guided_bp': AttributionMethods.GuidedBackprop,
        'shapley': AttributionMethods.Shapley,
        'lime': AttributionMethods.LIME,
        'occlusion': AttributionMethods.Occlusion
    }

    parser = argparse.ArgumentParser(description='Run ERASER experiments')
    parser.add_argument('--output', choices=list(output_methods.keys()), help="Output mechanism")
    parser.add_argument('--agg', choices=list(aggregation_methods.keys()), help="Aggregation method")
    parser.add_argument('--model', type=str, help='Path to finetuned model')

    parser.add_argument('--method', choices=list(attr_methods.keys()), help='Attribution method')
    parser.add_argument('--device', type=str, help='device')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size used for attribution calculation, '
                                                                   'automatically set to 1 for some methods regardless '
                                                                   'of choice')
    parser.add_argument('--output-dir', type=str, default='logs', help='Path to directory to save results')
    parser.add_argument('--n-steps', type=int, help='IntegratedGradients number of steps')

    args = parser.parse_args()

    nli_model_name = os.path.abspath(args.model)

    attr_config = AttributionConfig(attr_methods[args.method], remove_pad_tokens=True,
                                    remove_cls_token=True, remove_sep_tokens=False, join_subwords=True,
                                    normalize_scores=True,
                                    forward_scoring=output_methods[args.output],
                                    aggregation_method=aggregation_methods[args.agg],
                                    label_names=['entailment', 'neutral', 'contradiction'])

    bs = args.batch_size
    output_dir = args.output_dir

    attribution = NLIAttribution(model_name=nli_model_name, config=attr_config, device=args.device)

    if args.method in ['shapley', 'lime']:
        comprehensiveness_score, sufficiency_score, all_compre_scores, all_suff_scores, compre_pairs_w_perturbations, \
        suff_pairs_w_perturbations, compre_pred_out, \
        suff_pred_out = eraser_faithfulness.evaluate(attribution,
                                                     percentages=[0.01, 0.05, 0.1, 0.2, 0.5],
                                                     batch_size=64, attr_batch_size=1,
                                                     return_all_outputs=True)
    elif args.method == 'ig':
        comprehensiveness_score, sufficiency_score, all_compre_scores, all_suff_scores, compre_pairs_w_perturbations, \
        suff_pairs_w_perturbations, compre_pred_out, \
        suff_pred_out = eraser_faithfulness.evaluate(attribution,
                                                     percentages=[0.01, 0.05, 0.1, 0.2, 0.5],
                                                     batch_size=64, attr_batch_size=1,
                                                     return_all_outputs=True, n_steps=args.n_steps)
    elif args.method == 'occlusion':
        comprehensiveness_score, sufficiency_score, all_compre_scores, all_suff_scores, compre_pairs_w_perturbations, \
        suff_pairs_w_perturbations, compre_pred_out, \
        suff_pred_out = eraser_faithfulness.evaluate(attribution,
                                                     percentages=[0.01, 0.05, 0.1, 0.2, 0.5],
                                                     batch_size=64, attr_batch_size=bs,
                                                     return_all_outputs=True, sliding_window_shapes=(1, 768))
    else:
        comprehensiveness_score, sufficiency_score, all_compre_scores, all_suff_scores, compre_pairs_w_perturbations, \
        suff_pairs_w_perturbations, compre_pred_out, \
        suff_pred_out = eraser_faithfulness.evaluate(attribution,
                                                     percentages=[0.01, 0.05, 0.1, 0.2, 0.5],
                                                     batch_size=64, attr_batch_size=bs,
                                                     return_all_outputs=True)

    save_results(attribution, nli_model_name, comprehensiveness_score, sufficiency_score)



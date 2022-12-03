from explainli.config import AttributionMethods, AttributionConfig, AggregationMethods, ForwardScoringOptions
from explainli.explainli import NLIAttribution
from eval import crosslingual_faithfulness

import pandas as pd
import numpy as np

import json
import argparse
import os
from typing import List, Tuple, Dict, Optional


def save_results(attribution: NLIAttribution, nli_model_name: str, avg_correlation: float,
                 avg_correlations_per_lang: Dict[str, float], avg_pval: float, avg_pvalues_per_lang: Dict[str, float],
                 output_dir: str) -> None:
    """
    Save average crosslingual faithfulness scores and related p-values along with results per language

    :param attribution:
    :param nli_model_name:
    :param avg_correlation:
    :param avg_correlations_per_lang
    :param avg_pval
    :param avg_pvalues_per_lang
    :param output_dir:
    """
    attr_method = getattr(attribution.config, 'attribution_method').name
    agg_method = getattr(attribution.config, 'aggregation_method').name.lower()
    output_mechanism = getattr(attribution.config, 'forward_scoring').name.lower()
    model_name = nli_model_name.split('/')[-1]

    data = {
        'correlation': {
            'avg_correlation': avg_correlation,
            'avg_correlations_per_lang': avg_correlations_per_lang
        },
        'pval': {
            'avg_pval': avg_pval,
            'avg_pvals_per_lang': avg_pvalues_per_lang
        },
    }

    with open(f'{output_dir}/{attr_method}_{agg_method}_wrt_{output_mechanism}_{model_name}.json', 'w') as log_file:
        json.dump(data, log_file, indent=6)


def load_dataset_with_alignments(fname: str) -> Tuple[Dict[str, List[Tuple[str, str]]], List[str],
                                                      List[Tuple[List[int], List[int]]]]:
    """
    Load pre-extracted alignments from given file

    :param fname
    :return
    """
    with open(fname, 'r') as f:
        data_as_dict = json.load(f)
        alignments = list(
            map(lambda x: set(map(lambda y: tuple(y), x)) if x is not None else None, data_as_dict['alignments']))
        pairs, labels = data_as_dict['pairs'], data_as_dict['labels']

    return pairs, labels, alignments


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

    parser = argparse.ArgumentParser(description='Run crosslingual faithfulness experiments')
    parser.add_argument('--output', choices=list(output_methods.keys()), help="Output mechanism")
    parser.add_argument('--agg', choices=list(aggregation_methods.keys()), help="Aggregation method")
    parser.add_argument('--model', type=str, help='Path to finetuned model')
    parser.add_argument('--alignments', type=str, help='Path to alignments')
    parser.add_argument('--alignments-set', choices=['best', 'worst'], help='Type of alignments whether it is for best '
                                                                            'or worst performing set of languages')
    parser.add_argument('--method', choices=list(attr_methods.keys()), help='Attribution method')
    parser.add_argument('--device', type=str, help='device')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size used for attribution calculation, '
                                                                   'automatically set to 1 for some methods regardless '
                                                                   'of choice')
    parser.add_argument('--output-dir', type=str, default='logs', help='Path to directory to save results')
    parser.add_argument('--n-steps', type=int, help='IntegratedGradients number of steps')

    args = parser.parse_args()

    nli_model_name = os.path.abspath(args.model)

    ds = load_dataset_with_alignments(os.path.abspath(args.alignments))

    attr_config = AttributionConfig(attr_methods[args.method], remove_pad_tokens=True,
                                    remove_cls_token=True, remove_sep_tokens=True, join_subwords=True,
                                    normalize_scores=True,
                                    forward_scoring=output_methods[args.output],
                                    aggregation_method=aggregation_methods[args.agg],
                                    label_names=['entailment', 'neutral', 'contradiction'])

    if args.alignments_set == 'best':
        languages = ['en', 'es', 'fr', 'de', 'bg']
    elif args.alignments_set == 'worst':
        languages = ['en', 'ur', 'th', 'sw']

    bs = args.batch_size
    output_dir = args.output_dir

    attribution = NLIAttribution(model_name=nli_model_name, config=attr_config, device=args.device)

    if args.method in ['shapley', 'lime']:
        ds, avg_correlation, avg_correlations_per_lang, avg_pval, avg_pvalues_per_lang = crosslingual_faithfulness.evaluate(
            attribution, 1, dataset_with_alignments=ds, selected_languages=languages)
    elif args.method == 'ig':
        ds, avg_correlation, avg_correlations_per_lang, avg_pval, avg_pvalues_per_lang = crosslingual_faithfulness.evaluate(
            attribution, 1, dataset_with_alignments=ds, selected_languages=languages, n_steps=args.n_steps)
    elif args.method == 'occlusion':
        ds, avg_correlation, avg_correlations_per_lang, avg_pval, avg_pvalues_per_lang = crosslingual_faithfulness.evaluate(
            attribution, bs, sliding_window_shapes=(1, 768), dataset_with_alignments=ds, selected_languages=languages)
    else:
        ds, avg_correlation, avg_correlations_per_lang, avg_pval, avg_pvalues_per_lang = crosslingual_faithfulness.evaluate(
            attribution, 16, dataset_with_alignments=ds, selected_languages=languages)

    save_results(attribution, nli_model_name, avg_correlation, avg_correlations_per_lang, avg_pval, avg_pvalues_per_lang, args.output_dir)


from eval import human_agreement
from explainli.config import AttributionMethods, AttributionConfig, AggregationMethods, ForwardScoringOptions
from explainli.explainli import NLIAttribution

import pandas as pd
import numpy as np

import json
import argparse
import os


def save_results(attribution: NLIAttribution, nli_model_name: str, map_score: float, f1: float, output_dir: str,
                 n: int = None, split: str = None) -> None:
    """
    Save MAP and F1 scores

    :param attribution:
    :param nli_model_name:
    :param map_score:
    :param f1:
    :param n:
    :param split:
    """
    attr_method = getattr(attribution.config, 'attribution_method').name
    agg_method = getattr(attribution.config, 'aggregation_method').name.lower()
    output_mechanism = getattr(attribution.config, 'forward_scoring').name.lower()
    model_name = nli_model_name.split('/')[-1]

    data = {
        'map': map_score,
        'f1': f1
    }
    nsteps_str = f"_n_{n}" if n is not None and split == 'dev' else ""
    with open(f'{output_dir}/{attr_method}_{agg_method}_wrt_{output_mechanism}_{model_name}{nsteps_str}.json',
              'w') as log_file:
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

    parser = argparse.ArgumentParser(description='Run human agreement experiments.')
    parser.add_argument('--output', choices=list(output_methods.keys()), help="Output mechanism")
    parser.add_argument('--agg', choices=list(aggregation_methods.keys()), help="Aggregation method")
    parser.add_argument('--method', choices=list(attr_methods.keys()), help='Attribution method')
    parser.add_argument('--model', type=str, help='Path to finetuned model')
    parser.add_argument('--dataset-dir', type=str, help='Path to the directory in which the e-SNLI dataset exists')
    parser.add_argument('--split', choices=['dev', 'test'], help='dev/test split')
    parser.add_argument('--device', type=str, help='device')
    parser.add_argument('--batch-size', type=int, default=192, help='Batch size used for attribution calculation, '
                                                                   'automatically set to 1 for some methods regardless '
                                                                   'of choice')
    parser.add_argument('--output-dir', type=str, default='logs', help='Path to directory to save results')
    parser.add_argument('--n-steps', type=int, help='IntegratedGradients number of steps')

    args = parser.parse_args()

    nli_model_name = os.path.abspath(args.model)

    attr_config = AttributionConfig(attr_methods[args.method], remove_pad_tokens=True,
                                    remove_cls_token=True, remove_sep_tokens=True, join_subwords=True,
                                    normalize_scores=True,
                                    forward_scoring=output_methods[args.output],
                                    aggregation_method=aggregation_methods[args.agg],
                                    label_names=['entailment', 'neutral', 'contradiction'])

    bs = args.batch_size
    output_dir = args.output_dir
    dataset_path = os.path.abspath(f'{args.dataset_dir}/esnli_{args.split}.csv')
    threshold = 0.5 if args.agg == 'l2' else 0.0

    attribution = NLIAttribution(model_name=args.model, config=attr_config, device=args.device)


    if args.method in ['shapley', 'lime']:
        _, mAP, f1 = human_agreement.evaluate(attribution, dataset_path, 1, threshold)
    elif args.method == 'ig':
        _, mAP, f1 = human_agreement.evaluate(attribution, dataset_path, 1, threshold, n_steps=args.n_steps)
    elif args.method == 'occlusion':
        _, mAP, f1 = human_agreement.evaluate(attribution, dataset_path, bs, threshold, sliding_window_shapes=(1, 768))
    else:
        _, mAP, f1 = human_agreement.evaluate(attribution, dataset_path, bs, threshold)
    print(f"mAP: {mAP}, f1: {f1}")
    save_results(attribution, args.model, mAP, f1, split=args.split, n=args.n_steps)

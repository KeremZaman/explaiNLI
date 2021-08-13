from explainli.explainli import NLIAttribution
from captum.attr import visualization as viz

from transformers import pipeline
from datasets import load_dataset
from tqdm import tqdm

from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import Dict, Union, List, Tuple, Optional


def create_dataset_perturbations(data: Dict[str, Union[str, int]], unmasker: pipeline, tokenizer: PreTrainedTokenizer,
                                 num_sample: int) -> Tuple[List[Tuple[str, str]], List[int]]:
    """
    This function creates required perturbations for statistical significance calculation. Firstly, it generates new
    sentence pairs by masking each token, then  it generates as many  pairs as K(num_sample) for each masked sentence
    by replacing each mask token with a possible token via MLM

    Perturbed data format:

    w_1 w_2 w_3 w_4 ... <orig sentence_1>
    w_11 w_2 w_3 w_4 ... <sampled sentence for w_1>
    w_12 w_2 w_3 w_4 ... <sampled sentence for w_1>
    w_13 w_2 w_3 w_4 ... <sampled sentence for w_1>
    .    .   .   .
    .    .   .   .
    .    .   .   .
    w_1k w_2 w_3 w_4 ... <last sampled sentence for w_1>

    w_1 w_2 w_3 w_4 ... <orig sentence_1>
    w_1 w_21 w_3 w_4 ... <sampled sentence for w_2>
    w_1 w_22 w_3 w_4 ... <sampled sentence for w_2>
    w_1 w_23 w_3 w_4 ... <sampled sentence for w_2>
    .   .    .   .
    .   .    .   .
    .   .    .   .
    w_1 w_2k w_3 w_4 ... <last sampled sentence for w_2>
    .
    .
    w_1 w_2 w_3 w_4 ... w_nk <last sampled sentence for w_n>
    ...
    w_1 w_2 w_3 w_4 ... <orig sentence_2>
    ...
    w_1 w_2 w_3 w_4 ... w_nk <last sampled sentence for w_n>
    ...

    :param data:
    :param unmasker:
    :param tokenizer:
    :param num_sample:
    :return:
    """
    pairs, labels = [], []

    # tokenize all data at first
    premise_tokens_list = tokenizer(data['premise'])['input_ids']
    hypothesis_tokens_list = tokenizer(data['hypothesis'])['input_ids']
    hypothesis_tokens_list.pop(0)  # remove CLS token
    label_list = data['label']

    print("Perturbing data...")
    for premise_tokens, hypothesis_tokens, label in tqdm(zip(premise_tokens_list, hypothesis_tokens_list, label_list)):

        # combine premise and hypothesis so that it becomes: [CLS] <premise_tokens> [SEP] <hypothesis_tokens> [SEP]
        sentence_tokens = premise_tokens + hypothesis_tokens

        sentence = tokenizer.decode(sentence_tokens)

        # remove CLS and SEP for creating pairs because attribution object will add them during tokenization
        pair = tuple(sentence.replace(tokenizer.cls_token, '').strip().split(tokenizer.sep_token)[:2])

        # add original sentence as pair and save its label
        pairs.append(pair)
        labels.append(label)

        masked_sentence_tokens_list = []

        # create new sentences by masking each token separately
        for i, token in enumerate(sentence_tokens):
            # do not apply mask for special tokens
            if token in [tokenizer.sep_token_id, tokenizer.cls_token_id]:
                continue

            masked_sentence_tokens = sentence_tokens[:]
            masked_sentence_tokens[i] = tokenizer.mask_token_id
            masked_sentence_tokens_list.append(masked_sentence_tokens)

        masked_sentences = [tokenizer.decode(tokens) for tokens in masked_sentence_tokens_list]

        # sample new tokens (top-k) for the masked ones
        sampled_tokens = [[sample['token_str'] for sample in samples] for samples in
                          unmasker(masked_sentences, top_k=num_sample)]

        for masked_sentence, sentence_samples in zip(masked_sentences, sampled_tokens):
            for token in sentence_samples:
                # replace masked token with the sampled token for each masked sentence
                unmasked_sentence = masked_sentence.replace(tokenizer.mask_token, token)
                # add new pair for sampled sentence for calculating attribution scores later
                pair = tuple(unmasked_sentence.replace(tokenizer.cls_token, '').strip().split(tokenizer.sep_token)[:2])
                pairs.append(pair)
                labels.append(label)

    return pairs, labels


def _calculate_attributions(pairs: List[Tuple[str, str]], labels: List[int], attribution: NLIAttribution,
                            batch_size: int, **kwargs):
    """
    Calculate attribution scores for each instance by splitting into batches

    :param pairs:
    :param labels:
    :param attribution:
    :param batch_size:
    :param kwargs:
    :return:
    """
    for i in tqdm(range(0, len(pairs), batch_size)):
        attribution.attr(pairs[i:i + batch_size], labels[i:i + batch_size], **kwargs)


def _calculate_pvalues(attribution: NLIAttribution, num_samples: int) -> List[Tuple[List[str], List[float], str, str]]:
    """
    Calculate p-values from sampled sentence attributions for each word

    :param attribution:
    :param num_samples:
    :return:
    """
    p_values_with_sentences = []

    i = 0
    start, end = 0, 0

    while i < len(attribution.records):
        record = attribution.records[i]
        num_words = len(record.word_attributions)
        word_p_values = []

        for j in range(num_words):
            orig_score = record.word_attributions[j]
            start = i + j * num_samples + 1
            end = start + num_samples
            pval = (1 + sum([1 if orig_score <= rec.word_attributions[j] else 0 for rec in
                             attribution.records[start:end]])) / (num_samples + 1)
            word_p_values.append(pval)

        p_values_with_sentences.append((record.raw_input, word_p_values, record.pred_class, record.true_class))
        i = end

    return p_values_with_sentences


def calculate_significance(p_values_with_sentences: List[Tuple[List[str], List[float]]], alpha: float):
    """
    Calculate whether a token is significant or not considering given significance level, return ratio of
    statistically significant tokens to all tokens and visualization records for statistical significance

    :param p_values_with_sentences:
    :param alpha:
    :return:
    """
    num_total_words, num_significant_words = 0, 0
    viz_records = []

    for words, p_values, pred_label, true_label in p_values_with_sentences:

        significancy_list = [1 if p <= alpha else 0 for p in p_values]

        num_total_words += len(significancy_list)
        num_significant_words += sum(significancy_list)

        record = viz.VisualizationDataRecord(significancy_list, 0, pred_label, true_label, "",
                                             sum(significancy_list) / len(significancy_list), words, 0.0)
        viz_records.append(record)

    significancy_ratio = num_significant_words / num_total_words

    return significancy_ratio, viz_records


def visualize_significant_words(viz_records, start: Optional[int] = 0, num: Optional[int] = None):
    """
    Visualize statistically significant words from given visualization records

    :param viz_records:
    :param start:
    :param num:
    :return:
    """
    end = start + num if num is not None else len(viz_records)
    viz.visualize_text(viz_records[start:end])


def evaluate(attribution: NLIAttribution, dataset_name: str, sampling_model_name: str, num_samples: int,
             batch_size: int, alpha: float, perturbed_dataset: Optional[Tuple[List[str], List[int]]]=None,
             max_instances: Optional[int]=None, **kwargs):
    """
    Perform statistical significance calculation on given raw dataset/perturbed dataset

    :param attribution:
    :param dataset_name: Name of dataset to be used from HF datasets
    :param sampling_model_name: The name of pretrained HF model to be used to perform fill-mask task
    :param num_samples: Number of words to be sampled for each masked word
    :param batch_size:
    :param alpha: significance level
    :param perturbed_dataset: dataset that is already perturbed for p-value calculation
    :param max_instances: max number of instances to be evaluated from raw dataset
    :param kwargs:
    :return:
    """
    if perturbed_dataset:
        pairs, labels = perturbed_dataset
    else:
        dataset = load_dataset(dataset_name, split='test')

        if max_instances:
            dataset = dataset[:max_instances]

        unmasker = pipeline('fill-mask', model=sampling_model_name)
        tokenizer = AutoTokenizer.from_pretrained(sampling_model_name)

        pairs, labels = create_dataset_perturbations(dataset, unmasker, tokenizer, num_samples)
        perturbed_dataset = pairs, labels

    _calculate_attributions(pairs, labels, attribution, batch_size, **kwargs)

    pvalues = _calculate_pvalues(attribution, num_samples)

    significancy_ratio, viz_records = calculate_significance(pvalues, alpha)

    return significancy_ratio, viz_records, perturbed_dataset



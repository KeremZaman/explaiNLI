from explainli.explainli import NLIAttribution
from eval.human_agreement import read_dataset, calculate_attributions, _create_gold_labels, calculate_map
from eval.crosslingual_faithfulness import awesome_align, _word_tokenize

from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer
import datasets

from sklearn.metrics import precision_recall_curve
import numpy as np
from tqdm import tqdm
import pandas as pd

from typing import Tuple, Dict, List, Optional


def find_threshold_on_esnli(attribution: NLIAttribution, dataset_path: str, batch_size: int, **kwargs) -> Tuple[float,
                                                                                                                float]:
    """
    Find best threshold for attribution scores on e-SNLI dataset for the given attribution method

    :param attribution:
    :param dataset_path: path to e-SNLI dataset
    :param batch_size:
    :param kwargs:
    :return:
    """
    data = read_dataset(dataset_path, attribution.label_names)
    calculate_attributions(data, attribution, batch_size, **kwargs)

    gold_labels = [label for item in data for label in item['highlight_gold_labels']]
    preds = [score for record in attribution.records for score in record.word_attributions]

    precision, recall, thresholds = precision_recall_curve(gold_labels, preds)
    f1_scores = 2*recall*precision/(recall + precision)

    best_f1, best_threshold = np.max(f1_scores), thresholds[np.argmax(f1_scores)]

    return best_f1, best_threshold


def extract_rationales(attribution: NLIAttribution, dataset: datasets.DatasetDict, threshold: float, batch_size: int,
                       src_lang: Optional[str] = 'en', **kwargs) -> \
        Tuple[Dict[str, List[Tuple[str, str]]], List[str], List[Tuple[List[int], List[int]]]]:
    """
    Extract rationales from XNLI English split according to binarized attribution scores based on given threshold

    NOTE: Please set `remove_sep_tokens` to False for attribution configuration, since SEP tokens are used to
    reconstruct premise-hypothesis pairs from preprocessed input

    :param attribution:
    :param dataset: HF dataset object for XNLI dataset
    :param threshold:
    :param batch_size:
    :param src_lang
    :param kwargs:
    :return:
    """

    languages = dataset['hypothesis'][0]['language']
    pairs = {lang: [] for lang in languages}
    labels = []

    # extract pairs and labels and group them wrt languages
    for premises, hypotheses, label in tqdm(zip(dataset['premise'], dataset['hypothesis'], dataset['label'])):
        for lang in languages:
            lang_idx = languages.index(lang)
            premise = premises[lang]
            hypothesis = hypotheses['translation'][lang_idx]
            pairs[lang].append((premise, hypothesis))
            labels.append(label)

    # calculate attribution scores for source language split
    for i in tqdm(range(0, len(pairs[src_lang]), batch_size)):
        attribution.attr(pairs[src_lang][i:i + batch_size], labels[i:i + batch_size], **kwargs)

    all_highlight_idxs = []

    for record in attribution.records:
        # find sep token position to split combined text into premise and hypothesis again
        # do not include sep token
        sep_pos = record.raw_input.index(attribution.tokenizer.sep_token)
        prem_attributions, hyp_attributions = np.array(record.word_attributions[:sep_pos]), \
                                              np.array(record.word_attributions[sep_pos+1:])

        # get positions of the words having score bigger than threshold
        highlight_idxs = np.where(prem_attributions > threshold)[0].tolist(), np.where(hyp_attributions >
                                                                                       threshold)[0].tolist()
        all_highlight_idxs.append(highlight_idxs)

    return pairs, labels, all_highlight_idxs


def align_rationales(pairs: Dict[str, List[Tuple[str, str]]], labels: List[str],
                     src_highlight_idxs: List[Tuple[List[int], List[int]]], word_aligner: Optional[str],
                     src_lang: Optional[str] = 'en') -> \
        Tuple[Dict[str, List[Tuple[str, str]]], List[str], Dict[str, List[Tuple[List[int], List[int]]]]]:
    """
    Extract word alignments between source (English) and target languages (other XNLI languages), use these alignments
    to find corresponding rationales of the extracted rationales for the source language in target languages
    :param pairs:
    :param labels:
    :param src_highlight_idxs:
    :param word_aligner:
    :param src_lang:
    :return:
    """

    # initialize multilingual model and tokenizer for awesome-align
    multilingual_model_name = 'bert-base-multilingual-cased' if word_aligner is None else word_aligner
    tokenizer = AutoTokenizer.from_pretrained(multilingual_model_name)
    model = AutoModel.from_pretrained(multilingual_model_name)

    languages = list(pairs.keys())
    # remove source language from list of languages
    languages.remove(src_lang)
    translated_highlight_idxs = {lang: [] for lang in languages}

    for lang in languages:
        for i, (pair, label, highlight_idxs) in tqdm(enumerate(zip(pairs[lang], labels, src_highlight_idxs))):
            premise, hypothesis = pair
            src_premise, src_hypothesis = pairs[src_lang][i]
            src_prem_highlight_idxs, src_hyp_highlight_idxs = highlight_idxs

            prem_highlights, hyp_highlights = [], []

            # calculate alignment between words in target and source premises
            premise_alignments = awesome_align(src_premise, premise, tokenizer, model)
            # find corresponding position of each rationale (if they have) and add them to list of rationales for target
            for j, k in premise_alignments:
                if j in src_prem_highlight_idxs:
                    prem_highlights.append(k)

            # calculate alignment between words in target and source hypotheses
            hypothesis_alignments = awesome_align(src_hypothesis, hypothesis, tokenizer, model)
            # find corresponding position of each rationale (if they have) and add them to list of rationales for target
            for j, k in hypothesis_alignments:
                if j in src_hyp_highlight_idxs:
                    hyp_highlights.append(k)

            translated_highlight_idxs[lang].append((prem_highlights, hyp_highlights))

    # add rationales for source language to collect all highlight (rationale) positions into one dict
    translated_highlight_idxs[src_lang] = src_highlight_idxs

    return pairs, labels, translated_highlight_idxs


def create_dataset(path: str, attribution: NLIAttribution, pairs: Dict[str, List[Tuple[str, str]]], labels: List[str],
                   all_highlights: Dict[str, List[Tuple[List[int], List[int]]]], tokenizer: PreTrainedTokenizer):
    """
    Create CSV file for new dataset in similar format to e-SNLI
    :param path: path to CSV file to save multilingual
    :param attribution
    :param pairs:
    :param labels:
    :param all_highlights:
    :param tokenizer:
    :return:
    """
    dataset = []
    languages = list(pairs.keys())
    label_names = attribution.label_names

    for lang in languages:
        for pair, label, highlights in zip(pairs[lang], labels, all_highlights[lang]):
            prem, hyp = pair
            prem_words, hyp_words = _word_tokenize(prem, tokenizer), _word_tokenize(hyp, tokenizer)
            prem_highlights, hyp_highlights = highlights
            highlighted_prem = ' '.join([f"*{word}*" if i in prem_highlights else word for i, word in enumerate(prem_words)])
            highlighted_hyp = ' '.join([f"*{word}*" if i in hyp_highlights else word for i, word in enumerate(hyp_words)])

            dataset.append([lang, label_names[label], prem, hyp, highlighted_prem, highlighted_hyp])

    data = pd.DataFrame(data=dataset, columns=['language', 'label', 'premise', 'hypothesis', 'premise_highlighted',
                                               'hypothesis_highlighted'])
    data.to_csv(path, index=False, encoding='utf-8')


def evaluate_multilingual_plausibility(dataset: str, attribution: NLIAttribution, languages: List[str],
                                       batch_size: int, **kwargs) -> Dict[str, float]:
    """
    Calculate MAP score for each language to measure plausibility
    :param dataset: path to CSV file for multilingual highlights
    :param attribution:
    :param batch_size
    :param languages: list of languages to measure plausbility
    :return:
    """

    df = pd.read_csv(dataset)
    label_names = attribution.label_names

    map_scores = {}

    for lang in languages:
        # convert csv file to required format for map score to be measured for already existing methods
        data = []
        print(lang)
        for index, row in df[df.language == lang].iterrows():
            gold_labels = _create_gold_labels(row['premise_highlighted']) + _create_gold_labels(row['hypothesis_highlighted'])
            premise, hypothesis, label = row['premise'],  row['hypothesis'], row['label']

            data.append({'id': index,
                         'sent1': premise,
                         'sent2': hypothesis,
                         'label': label_names.index(label),
                         'highlight_gold_labels': gold_labels})

        calculate_attributions(data, attribution, batch_size, **kwargs)
        map_score = calculate_map(data, attribution)
        map_scores[lang] = map_score
        print(f"MAP score for {lang}: {map_score}")

        attribution.records.clear()

    return map_scores

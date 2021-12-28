from explainli.explainli import NLIAttribution
from sklearn.metrics import average_precision_score, f1_score
from tqdm import tqdm
import numpy as np

from string import punctuation
from functools import reduce
import csv

from typing import List, Dict, Union, Tuple


def _create_gold_labels(highlighted_sentence: str) -> List[int]:
    """
    Create a list of gold labels from the given highlighted sentence

    :param highlighted_sentence:
    :return:
    """
    gold_labels = []

    for i, word in enumerate(highlighted_sentence.split()):

        # highlighted words are marked as *word*
        marked = True if (word.startswith('*') and word.endswith('*')) else False
        word = word.strip('*')

        # add space for each punctuation, split word to count punctuations as new tokens and make their label the same
        # with the main word
        word = reduce(lambda w, p: w.replace(p, f' {p} '), punctuation, word)
        for token in word.split():
            gold_labels.append(1 if marked else 0)

    return gold_labels


def read_dataset(path: str, label_names: List[str]) -> List[Dict[str, Union[int, str, List[int]]]]:
    """
    Read CSV file for e-SNLI dataset and convert to a dictionary consisting of pair ID, sentence pairs, label and a list
    of binary labels for highlights

    :param path: path to csv file
    :param label_names: names of NLI labels in order to convert them ids that are compatible with the model's ids
    :return:
    """
    data = []

    with open(path, newline='\n') as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='"')
        next(reader)  # pass header line

        for items in reader:
            id, label, sentence1, sentence2, highlights1, highlights2 = items[0], items[1], items[2], items[3], \
                                                                        items[5], items[6]

            gold_labels = _create_gold_labels(highlights1) + _create_gold_labels(highlights2)

            data.append({'id': id,
                         'sent1': sentence1,
                         'sent2': sentence2,
                         'label': label_names.index(label),
                         'highlight_gold_labels': gold_labels})
    return data


def calculate_attributions(data: List[Dict[str, Union[int, str, List[int]]]], attribution: NLIAttribution,
                           batch_size: int, **kwargs):
    """
    Calculate attribution scores for each instance in the given dataset by splitting into batches

    :param data:
    :param attribution:
    :param batch_size:
    :param kwargs:
    :return:
    """
    for i in tqdm(range(0, len(data), batch_size)):
        pairs = []
        labels = []

        for item in data[i:i + batch_size]:
            pairs.append((item['sent1'], item['sent2']))
            labels.append(item['label'])

        attribution.attr(pairs, labels, **kwargs)


def calculate_map(data: List[Dict[str, Union[int, str, List[int]]]], attribution: NLIAttribution) -> float:
    """
    Calculate mean Average Precision (mAP) by using attribution scores and gold labels for each instance

    :param data:
    :param attribution:
    :return:
    """
    ap_scores = []

    for record, item in zip(attribution.records, data):
        ap = average_precision_score(item['highlight_gold_labels'], record.word_attributions)
        ap_scores.append(ap)

    mean_ap = np.mean(ap_scores)

    return mean_ap


def calculate_f1(data: Dict[str, Union[int, str, List[int]]], attribution: NLIAttribution, threshold: float=0.5) -> float:
    """
    Calculate F1 score by using attribution scores and gold labels for each instance

    :param data:
    :param attribution:
    :param threshold:
    :return:
    """

    gold_labels = [label for item in data for label in item['highlight_gold_labels']]
    pred = [1 if word_score >= threshold else 0 for record in attribution.records for word_score in record.word_attributions]

    f1 = f1_score(gold_labels, pred)

    return f1


def evaluate(attribution: NLIAttribution, dataset_path: str, batch_size: int, threshold: float=0.5, **kwargs) -> \
        Tuple[Dict, float, float]:
    """
    Evaluate agreement with human rationales by calculating mean average precision and f1 scores for the given dataset
    with the given attribution object

    :param attribution:
    :param dataset_path:
    :param batch_size:
    :param threshold:
    :param kwargs:
    :return:
    """

    data = read_dataset(dataset_path, attribution.label_names)
    calculate_attributions(data, attribution, batch_size, **kwargs)

    map_score = calculate_map(data, attribution)
    f1 = calculate_f1(data, attribution, threshold)

    return data, map_score, f1







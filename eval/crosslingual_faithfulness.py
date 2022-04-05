from explainli.explainli import NLIAttribution

from transformers import PreTrainedTokenizer, PreTrainedModel, AutoTokenizer, AutoModel, BertTokenizer, BertTokenizerFast
from datasets import load_dataset
import datasets

import torch
import numpy as np
from scipy.stats import spearmanr

from tqdm import tqdm

import itertools
from typing import Set, Tuple, List, Dict, Optional
import string


def _word_tokenize(txt: str, tokenizer: PreTrainedTokenizer) -> List[str]:
    """
    Split sentences into words in a way that is compatible with the given pretrained tokenizer which is joining subwords
    after tokenization

    :param txt:
    :return:
    """
    token_ids = tokenizer(txt, add_special_tokens=False)['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    i = 0
    while i < len(tokens):
        if tokens[i].startswith('##'):
            tokens[i] = tokens[i].replace('##', '')
            tokens[i - 1] = tokens[i - 1] + tokens[i]
            tokens.pop(i)
        else:
            i += 1

    return tokens


def awesome_align(src: str, tgt: str, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, align_layer: int = 8,
                  threshold: float = 1e-3) -> Set[Tuple[int, int]]:
    """
    awesome-align implementation using mBERT (https://arxiv.org/abs/2101.08231)
    The function is adapted from the demo notebook at https://github.com/neulab/awesome-align

    :param src:
    :param tgt:
    :param tokenizer:
    :param model:
    :param align_layer:
    :param threshold:
    :return:
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    # split to words in the same way with the tokenizer to prevent different interpretations of some unicode
    # characters such as \u001d, \u06d4
    src_sent, tgt_sent = _word_tokenize(src, tokenizer), _word_tokenize(tgt, tokenizer)

    src_word_ids = tokenizer(src_sent, add_special_tokens=False)['input_ids']
    tgt_word_ids = tokenizer(tgt_sent, add_special_tokens=False)['input_ids']

    src_ids = tokenizer.prepare_for_model(list(itertools.chain(*src_word_ids)), return_tensors='pt')['input_ids'].to(
        device)
    tgt_ids = tokenizer.prepare_for_model(list(itertools.chain(*tgt_word_ids)), return_tensors='pt')['input_ids'].to(
        device)

    sub2word_map_src = []
    for i, word_list in enumerate(src_word_ids):
        sub2word_map_src += [i for x in word_list]

    sub2word_map_tgt = []
    for i, word_list in enumerate(tgt_word_ids):
        sub2word_map_tgt += [i for x in word_list]

    out_src = model(src_ids.unsqueeze(0), output_hidden_states=True).hidden_states[align_layer][0, 1:-1]
    out_tgt = model(tgt_ids.unsqueeze(0), output_hidden_states=True).hidden_states[align_layer][0, 1:-1]

    dot_prod = out_src @ out_tgt.transpose(-1, -2)

    softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
    softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

    softmax_inter = (softmax_srctgt > threshold) * (softmax_tgtsrc > threshold)

    align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
    align_words = set()

    for i, j in align_subwords:
        align_words.add((sub2word_map_src[i], sub2word_map_tgt[j]))

    return align_words


def create_dataset_with_alignments(dataset: datasets.DatasetDict, word_aligner: Optional[str] = None,
                                   selected_languages: Optional[List[str]] = None) -> \
        Tuple[List[Tuple[str, str]], List[int], List[Set[Tuple[int, int]]]]:
    """
    Create dataset of pairs with their ground truth labels and word alignment maps.

    :param dataset: HF Dataset object
    :param word_aligner: Path to alignment model
    :param selected_languages: List of languages to process from the dataset, if given
    :return:
    """

    languages = list(dataset['premise'][0].keys())
    idx_en = dataset['hypothesis'][0]['language'].index('en')

    # set model and tokenizer for word alignment
    multilingual_model_name = 'bert-base-multilingual-cased' if word_aligner is None else word_aligner
    tokenizer = AutoTokenizer.from_pretrained(multilingual_model_name)
    model = AutoModel.from_pretrained(multilingual_model_name)

    pairs, labels, alignments = [], [], []

    for premise, hypothesis, label in tqdm(zip(dataset['premise'], dataset['hypothesis'], dataset['label'])):
        # insert English instances in first place
        en_premise = premise['en']
        en_hypothesis = hypothesis['translation'][idx_en]
        pairs.append((en_premise, en_hypothesis))
        labels.append(label)
        # no need to calculate alignment of an english instance with itself
        alignments.append(None)

        # add translations first, then continue from new instances

        for i, lang in enumerate(languages):
            # English instance is already inserted
            if lang.strip() == 'en':
                continue
            if selected_languages is not None and lang.strip() not in selected_languages:
                continue

            pairs.append((premise[lang], hypothesis['translation'][i]))
            labels.append(label)

            # combine premise and hypothesis for word alignment
            src_combined = f"{en_premise} {en_hypothesis}"
            tgt_combined = f"{premise[lang]} {hypothesis['translation'][i]}"
            alignment = awesome_align(src_combined, tgt_combined, tokenizer, model)
            alignments.append(alignment)

    return pairs, labels, alignments


def _convert_scores_for_alignment(words: List[str], scores: List[float], tokenizer: PreTrainedTokenizer,
                                  aligner_tokenizer: PreTrainedTokenizer) -> Tuple[List[str], List[float]]:
    """
    awesome-align uses a BERT-based tokenizer. When XLM-R is used for getting attribution scores, words may not have
    one-to-one correspondence between two tokenizers. Alignment mapping is extracted wrt BERT tokenizer, while scores
    are calculated wrt XLM-R tokenizer. Since BERT tokenizer produces more atomic tokens, words than XLM-R's, we split
    words using BERT tokenizer and divide scores accordingly to prevent mismatch between provided scores and alignment
    mapping.
    """

    if type(tokenizer) in [BertTokenizer, BertTokenizerFast]:
        return words, scores

    updated_words = []
    updated_scores = []

    for word, score in zip(words, scores):
        token_ids = aligner_tokenizer(word, add_special_tokens=False)['input_ids']
        tokens = aligner_tokenizer.convert_ids_to_tokens(token_ids)

        # BERT tokenizer can see some piece of texts as multiple words which XLM-R tokenizer sees as single word.
        # this is especially the case when the punctuation is considered. e.g. "flowers," can be a single word for
        # XLM-R while it's ["flow", "##er", "##s", ","] for BERT tokenizer (2 words, 4 tokens). For each word that
        # corresponds to multiple words for BERT tokenizer, we divide corresponding scores to the number of words
        # it corresponds to. For new words just consists of punctuation, we assign zero score.
        new_words = []
        new_word = ''
        for token in tokens:
            if not token.startswith('##') and new_word:
                new_words.append(new_word)
                new_word = token
            else:
                new_word += token.replace('##', '')

        new_words.append(new_word)

        n = len(new_words) - len(list(filter(lambda x: x in string.punctuation, new_words)))
        new_scores = [score / n if word not in string.punctuation else 0.0 for word in new_words]

        updated_words.extend(new_words)
        updated_scores.extend(new_scores)

    return updated_words, updated_scores


def calc_correlations(attribution: NLIAttribution, alignments: List[Set[Tuple[int, int]]], languages: List[str]) -> \
        Tuple[float, Dict[str, float], float, Dict[str, float]]:
    """

    Calculate average Spearman correlations and p-values of attribution scores of translated pairs wrt attribution
    scores of English ones.

    Attribution scores for different translations of a pair don't have to 1-to-1 correspondence with the English one, so
    take sum of the attribution scores of the corresponding words in target sequence for each word in source sequence

    :param attribution:
    :param alignments:
    :param languages:
    :return:
    """
    # put English in the beginning of list of languages
    languages.remove('en')
    languages.insert(0, 'en')

    num_languages = len(languages)
    correlations_per_lang = {lang: [] for lang in languages}
    pvalues_per_lang = {lang: [] for lang in languages}

    aligner_tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

    for i, (record, alignment) in enumerate(zip(attribution.records, alignments)):
        # set reference scores to attr scores for English
        if i % num_languages == 0:
            ref_scores = record.word_attributions
            ref_words = record.raw_input
            ref_words, ref_scores = _convert_scores_for_alignment(ref_words, ref_scores, attribution.tokenizer,
                                                                  aligner_tokenizer)
            continue

        scores = record.word_attributions
        words, scores = _convert_scores_for_alignment(record.raw_input, scores, attribution.tokenizer,
                                                      aligner_tokenizer)
        aligned_scores = [0 for i in range(len(ref_scores))]

        # sum the scores of corresponding words in target sequence for each word in source sequence to make two array
        # equal size
        for j, k in sorted(alignment):
            aligned_scores[j] += scores[k]

        # calculate spearman correlation and p-value and store them for each language
        rho, pval = spearmanr(ref_scores, aligned_scores)

        # rho/p-value can be NaN in some rare circumstances, ignore that instance in such cases
        if np.isnan(rho) or np.isnan(pval):
            continue

        correlations_per_lang[languages[i % num_languages]].append(rho)
        pvalues_per_lang[languages[i % num_languages]].append(pval)

    # average correlations for each language and across languages
    avg_correlations_per_lang = {}
    avg_pvalues_per_lang = {}
    for lang, correlations in correlations_per_lang.items():
        avg_correlations_per_lang[lang] = np.mean(correlations)
    for lang, pvalues in pvalues_per_lang.items():
        avg_pvalues_per_lang[lang] = np.mean(pvalues)

    # do not include English while averaging correlations and p-values since there is no value calculated for EN
    avg_correlation = np.mean(list(avg_correlations_per_lang.values())[1:])
    avg_pval = np.mean(list(avg_pvalues_per_lang.values())[1:])

    return avg_correlation, avg_correlations_per_lang, avg_pval, avg_pvalues_per_lang


def evaluate(attribution: NLIAttribution, batch_size: int, max_instances: Optional[int] = None,
             word_aligner: Optional[str] = None, selected_languages: Optional[List[str]] = None,
             dataset_with_alignments: Optional[Tuple] = None,
             **kwargs) -> Tuple[Tuple[List[Tuple[str, str]], List[int], List[Set[Tuple[int, int]]]], float,
                                Dict[str, float], float, Dict[str, float]]:
    """
    Evaluate faithfulness by looking at average correlations of attribution scores of XNLI translations wrt scores for
    Engish ones.

    Use mBERT fine-tuned on MNLI to get predictions and calculate attribution scores on XNLI dataset.

    This script uses awesome-align for word alignments which is required to align attribution scores while calculating
    correlations.

    :param attribution:
    :param batch_size:
    :param max_instances:
    :param word_aligner: path to alignment model
    :param selected_languages: List of languages to process from the dataset, if given
    :param dataset_with_alignments: dataset with pre-calculated alignments
    :param kwargs: additional arguments for attribution function
    :return:
    """

    # load XNLI dataset
    dataset = load_dataset('xnli', 'all_languages', split='test')

    if max_instances:
        dataset = dataset[:max_instances]

    pairs, labels, alignments = create_dataset_with_alignments(dataset, word_aligner, selected_languages) if \
        dataset_with_alignments is None else dataset_with_alignments

    # calculate attributions
    for i in tqdm(range(0, len(alignments), batch_size)):
        attribution.attr(pairs[i:i + batch_size], labels[i:i + batch_size], **kwargs)

    languages = list(dataset['premise'][0].keys())
    languages = list(filter(lambda x: x in selected_languages, languages)) if selected_languages else languages
    avg_correlation, avg_correlations_per_lang, avg_pval, avg_pvalues_per_lang = calc_correlations(attribution,
                                                                                                   alignments,
                                                                                                   languages)

    dataset_with_alignments = (pairs, labels, alignments)

    return dataset_with_alignments, avg_correlation, avg_correlations_per_lang, avg_pval, avg_pvalues_per_lang

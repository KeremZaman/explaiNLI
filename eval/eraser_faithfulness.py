from explainli.explainli import NLIAttribution

from torch.nn.functional import softmax
import torch

from transformers import PreTrainedModel, PreTrainedTokenizer
import datasets

import numpy as np
from tqdm import tqdm

from typing import List, Tuple, Optional


def get_predictions(pairs: List[Tuple[str, str]], model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                    batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Get all dataset, pass through the model in batches, combine outputs and return predictions for all pairs in given
    dataset

    :param pairs:
    :param model:
    :param tokenizer:
    :param batch_size:
    :param device:
    :return:
    """
    inputs = tokenizer(pairs, return_tensors='pt', padding=True).to(device)
    input_ids, token_type_ids, attention_mask = inputs['input_ids'], inputs.get('token_type_ids', None), \
                                                inputs['attention_mask']

    preds = []

    for i in tqdm(range(0, input_ids.shape[0], batch_size)):
        if token_type_ids is not None:
            output = model(input_ids=input_ids[:batch_size, :], token_type_ids=token_type_ids[:batch_size, :],
                       attention_mask=attention_mask[:batch_size, :])
        else:
            output = model(input_ids=input_ids[:batch_size, :], attention_mask=attention_mask[:batch_size, :])

        pred = softmax(output.logits, dim=1)
        preds.append(pred.detach().cpu())

    # shape: N x C
    # N: number of pairs C: number of classes
    return torch.vstack(preds)


def read_dataset(dataset: datasets.DatasetDict, language: Optional[str] = 'en') -> \
        Tuple[List[Tuple[str, str]], List[int]]:
    """

    :param dataset:
    :param language:
    :return:
    """
    languages = dataset['hypothesis'][0]['language']
    lang_idx = languages.index(language)

    pairs = []
    labels = []

    # extract pairs and labels and group them wrt languages
    for premises, hypotheses, label in tqdm(zip(dataset['premise'], dataset['hypothesis'], dataset['label'])):
        premise = premises[language]
        hypothesis = hypotheses['translation'][lang_idx]

        pairs.append((premise, hypothesis))
        labels.append(label)

    return pairs, labels


def calculate_comprehensiveness(attribution: NLIAttribution, percentages: List[float], batch_size: int) -> \
        Tuple[float, List[float], List[Tuple[str, str]], torch.Tensor]:
    """
    Comprehensiveness metric for faithfulness evaluation from DeYoung et al., 2020 https://arxiv.org/abs/1911.03429

    comprehensiveness = m(x_i)_j - m(x_i \ r_i)_j where m: model, j: class idx, x_i: input, r_i: rationales

    Deleting important tokens should decrease prediction probability which results in high comprehensiveness score.
    AOPC Comprehensiveness measures average of comprehensiveness when different percentages of tokens are deleted.

    \frac{1}{\B + 1} ( \sum_{k=0}^{| \B | } m(x_i)_j - m(x_i \ r_{ik})_j where r_{ik} is most important x tokens for
    k-th bin for i-th input.

    Say bins [0.01, 0.05, 0.1, 0.2, 0.5], then r_{ik} corresponds to top 1% tokens with respect to attribution scores
    when k = 1.

    :param attribution:
    :param percentages: ratio of most important tokens that each bin contains
    :param batch_size:
    :return:
    """
    # this list consists of original input pairs each followed by its perturbations
    pairs_w_perturbations = []
    compre_scores = []

    k = len(percentages)  # number of bins
    mask_token = attribution.tokenizer.mask_token
    sep_token = attribution.tokenizer.sep_token

    for record in attribution.records:
        # exclude end of sentence SEP token and corresponding attribution scores
        scores = record.word_attributions[:-1]
        words = record.raw_input[:-1]

        # sometimes attribution scores may contain nan values, which makes impossible to sort values
        # ignore instances when this rare event happens (it occurs when LIME is used with XLM-R)
        if np.isnan(scores).any():
            continue

        # some tokenizers (e.g xlm-r) can put two consecutive SEP tokens for text pairs unlike the common practice of
        # using single token. keep track of start and end position of SEP tokens to cover both cases
        sep_start_pos = words.index(sep_token)
        sep_end_pos = (sep_start_pos + 1) if words[sep_start_pos + 1] == sep_token else sep_start_pos

        # scores range between [0, 1], assign negative value to sep token
        # so that make sep token attribution smallest to prevent it from being erased and
        # to only use it for separating premise and hypothesis
        scores[sep_start_pos], scores[sep_end_pos] = -100, -100

        # insert original input in first place
        premise, hypothesis = ' '.join(words[:sep_start_pos]), ' '.join(words[sep_end_pos + 1:])
        pairs_w_perturbations.append((premise, hypothesis))

        # get scores and their indices in decreasing order
        sorted_idxs, sorted_scores = list(zip(*sorted(enumerate(scores), reverse=True, key=lambda x: x[1])))

        # convert percentages to integers for each instance
        # sometimes rounded number can be zero which means no token will be erased, set it to 1 in such cases
        # each integer represents number of tokens/words first k bins contain
        # e.g. percent_num_tokens = [2, 3, 7, 10] => k=1 bin consists of most important two tokens
        # k=2 bin consists of most important three tokens and so on.
        percent_num_tokens = [max(int(np.round(len(scores) * percentage)), 1) for percentage in percentages]

        # create perturbations by erasing top %x tokens
        for i, idx in enumerate(sorted_idxs):
            # deleting words at each step makes sorted_idxs useless, so replace the words to be deleted with MASK token
            # and delete when we hit the index of a bin
            words[idx] = mask_token

            # if we process number of tokens (i+1) that is enough to fill the next bin, insert perturbed input
            if (i+1) in percent_num_tokens:
                # create a perturbation by removing all words marked with mask token
                filtered_words = list(filter(lambda x: x != mask_token, words))

                sep_start_pos = filtered_words.index(sep_token)

                # find SEP tokens in perturbed instance
                # check if there is any element remains after first SEP token before checking there are two SEP tokens
                if (sep_start_pos + 1) < len(filtered_words) and filtered_words[sep_start_pos + 1] == sep_token:
                    sep_end_pos = sep_start_pos + 1
                else:
                    sep_end_pos = sep_start_pos

                # construct pair by splitting from SEP token in the middle
                premise, hypothesis = ' '.join(filtered_words[:sep_start_pos]), ' '.join(filtered_words[sep_end_pos + 1:])

                # for short sequences some percentages may correspond to same number of tokens (e.g. 5% of 10 words and
                # 10% of 10 words). when we hit a repeated number, insert the corresponding perturbation as many as
                # the number of bins it corresponds to
                for j in range(percent_num_tokens.count(i+1)):
                    pairs_w_perturbations.append((premise, hypothesis))

    # shape: N x C, N: number of all pairs C: number of classes
    outputs = get_predictions(pairs_w_perturbations, attribution.wrapper.model, attribution.tokenizer, batch_size,
                              attribution.device)

    for i in range(0, outputs.shape[0], k + 1):  # step size k+1 : number of perturbations + original input

        # get top prediction score and top predicted class index for original input
        original_pred, original_pred_class = torch.max(outputs[i], dim=0)

        start, end = i + 1, i + k + 1

        # prediction probabilities for the same class for perturbed inputs
        output_batch = outputs[start:end, original_pred_class]

        aopc_compre = (original_pred.repeat(1, k) - output_batch).sum() / (k + 1)
        compre_scores.append(aopc_compre)

    aopc_compre_score = np.mean(compre_scores)

    return aopc_compre_score, compre_scores, pairs_w_perturbations, outputs


def calculate_sufficiency(attribution: NLIAttribution, percentages: List[float], batch_size: int) -> \
        Tuple[float, List[float], List[Tuple[str, str]], torch.Tensor]:
    """
    Sufficiency metric for faithfulness evaluation from DeYoung et al., 2020 https://arxiv.org/abs/1911.03429

    sufficiency = m(x_i)_j - m(r_i)_j where m: model, j: class idx, x_i: input, r_i: rationales

    Just keeping important tokens should give similar prediction probability which results in low sufficiency score.
    AOPC Sufficiency measures average of sufficiency when different percentages of tokens are kept.

    \frac{1}{\B + 1} ( \sum_{k=0}^{| \B | } m(x_i)_j - m(r_{ik})_j where r_{ik} is most important x tokens for
    k-th bin for i-th input.

    Say bins [0.01, 0.05, 0.1, 0.2, 0.5], then r_{ik} corresponds to top 1% tokens with respect to attribution scores
    when k = 1.

    :param attribution:
    :param percentages:
    :param batch_size:
    :return:
    """
    # this list consists of original input pairs each followed by its perturbations
    pairs_w_perturbations = []
    suff_scores = []

    k = len(percentages)  # number of bins
    mask_token = attribution.tokenizer.mask_token
    sep_token = attribution.tokenizer.sep_token

    for record in attribution.records:
        # exclude end of sentence SEP token and corresponding attribution scores
        scores = record.word_attributions[:-1]
        words = record.raw_input[:-1]

        # sometimes attribution scores may contain nan values, which makes impossible to sort values
        # ignore instances when this rare event happens (it occurs when LIME is used with XLM-R)
        if np.isnan(scores).any():
            continue

        # for sufficiency, inputs consists of most important words which is more like addition than deletion
        # adding most important tokens one by one will lose the original order of the words
        # so replace all words with MASK token, later replace each MASK with the original word in the order of
        # decreasing importance. when all the remaining mask tokens are deleted, most important words will be left in
        # their original order

        # do not replace SEP with MASK, it's used for constructing pair
        sufficient_words = [mask_token if word != sep_token else sep_token for word in words]

        # some tokenizers (e.g xlm-r) can put two consecutive SEP tokens for text pairs unlike the common practice of
        # using single token. keep track of start and end position of SEP tokens to cover both cases
        sep_start_pos = words.index(sep_token)
        sep_end_pos = (sep_start_pos + 1) if words[sep_start_pos + 1] == sep_token else sep_start_pos

        # scores range between [0, 1], assign negative value to sep token
        # so that make sep token attribution smallest to prevent it from being erased and
        # to only use it for separating premise and hypothesis
        scores[sep_start_pos], scores[sep_end_pos] = -100, -100

        # insert original input
        premise, hypothesis = ' '.join(words[:sep_start_pos]), ' '.join(words[sep_end_pos + 1:])
        pairs_w_perturbations.append((premise, hypothesis))

        # get scores and their indices in decreasing order
        sorted_idxs, sorted_scores = list(zip(*sorted(enumerate(scores), reverse=True, key=lambda x: x[1])))

        # convert percentages to integers for each instance
        # sometimes rounded number can be zero which means no token will be erased, set it to 1 in such cases
        # each integer represents number of tokens/words first k bins contain
        # e.g. percent_num_tokens = [2, 3, 7, 10] => k=1 bin consists of most important two tokens
        # k=2 bin consists of most important three tokens and so on.
        percent_num_tokens = [max(int(np.round(len(scores) * percentage)), 1) for percentage in percentages]

        # create perturbations by keeping just top %x tokens
        for i, idx in enumerate(sorted_idxs):
            # replace MASKs with the original words by decreasing order of attribution scores
            # and delete remaining MASK when we hit the index of a bin to keep only important words
            sufficient_words[idx] = words[idx]

            # if we process number of tokens (i+1) that is enough to fill the next bin, insert perturbed input
            if (i + 1) in percent_num_tokens:
                # create a perturbation by removing all words marked with mask token to leave important words only
                filtered_words = list(filter(lambda x: x != mask_token, sufficient_words))

                sep_start_pos = filtered_words.index(sep_token)

                # find SEP tokens in perturbed instance
                # check if there is any element remains after first SEP token before checking there are two SEP tokens
                if (sep_start_pos + 1) < len(filtered_words) and filtered_words[sep_start_pos + 1] == sep_token:
                    sep_end_pos = sep_start_pos + 1
                else:
                    sep_end_pos = sep_start_pos

                # construct pair by splitting from SEP token in the middle
                premise, hypothesis = ' '.join(filtered_words[:sep_start_pos]), ' '.join(filtered_words[sep_end_pos + 1:])

                # for short sequences some percentages may correspond to same number of tokens (e.g. 5% of 10 words and
                # 10% of 10 words). when we hit a repeated number, insert the corresponding perturbation as many as
                # the number of bins it corresponds to
                for j in range(percent_num_tokens.count(i + 1)):
                    pairs_w_perturbations.append((premise, hypothesis))

    # shape: N x C, N: number of all pairs C: number of classes
    outputs = get_predictions(pairs_w_perturbations, attribution.wrapper.model, attribution.tokenizer, batch_size,
                              attribution.device)

    for i in range(0, outputs.shape[0], k + 1):  # step size k+1 : number of perturbations + original input

        # get top prediction score and top predicted class index for original input
        original_pred, original_pred_class = torch.max(outputs[i], dim=0)

        start, end = i + 1, i + k + 1

        # prediction probabilities for the same class for perturbed inputs
        output_batch = outputs[start:end, original_pred_class]

        aopc_suff = (original_pred.repeat(1, k) - output_batch).sum() / (k + 1)
        suff_scores.append(aopc_suff)

    aopc_suff_score = np.mean(suff_scores)

    return aopc_suff_score, suff_scores, pairs_w_perturbations, outputs


def evaluate(attribution: NLIAttribution, percentages: List[float], batch_size: int, language: Optional[str] = 'en',
             attr_batch_size: Optional[int] = None, return_all_outputs: Optional[bool] = False,
             dataset: Optional[datasets.DatasetDict] = None, **kwargs) -> \
        Tuple[float, float, Optional[List[float]], Optional[List[float]], Optional[List[Tuple[str, str]]],
              Optional[List[Tuple[str, str]]], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Evaluate faithfulness explanations with respect to sufficiency and comprehensiveness


    :param attribution:
    :param percentages: ratio of most important tokens that each bin contains
    :param batch_size:
    :param language: language from dataset that will be evaluated, default is English
    :param attr_batch_size: optional batch size for attribution calculation, some attribution methods may require
    smaller batch size than the one used for getting predictions from model
    :param return_all_outputs: if set True, return all intermediate comprehensiveness and sufficiency scores,
    all perturbed inputs and model predictions for them
    :param dataset: Optional Huggingface dataset which has the same format with XNLI
    :param kwargs:
    :return:
    """

    if dataset is None:
        dataset = datasets.load_dataset('xnli', 'all_languages', split='test')

    pairs, labels = read_dataset(dataset, language)

    # calculate attributions
    if attr_batch_size is None:
        attr_batch_size = batch_size

    for i in tqdm(range(0, len(pairs), attr_batch_size)):
        attribution.attr(pairs[i:i + attr_batch_size], labels[i:i + attr_batch_size], **kwargs)

    comprehensiveness_score, all_compre_scores, compre_pairs_w_perturbations, compre_pred_out = \
        calculate_comprehensiveness(attribution, percentages, batch_size)

    sufficiency_score, all_suff_scores, suff_pairs_w_perturbations, suff_pred_out = \
        calculate_sufficiency(attribution, percentages, batch_size)

    if return_all_outputs:
        return comprehensiveness_score, sufficiency_score, all_compre_scores, all_suff_scores, \
               compre_pairs_w_perturbations, suff_pairs_w_perturbations, compre_pred_out, suff_pred_out

    return comprehensiveness_score, sufficiency_score

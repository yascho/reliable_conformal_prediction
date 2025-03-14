import numpy as np
import torch.nn.functional as F
from scipy.special import softmax
from scipy.stats import binom
from tqdm.auto import tqdm

import time

from rcp.cp.score_functions import *


def conf_pred(test_idx, hparams, cal_indices, all_logits, targets, hashes):

    params = (test_idx, hparams, cal_indices, all_logits, targets, hashes)

    k_t = hparams['k_t']
    k_c = hparams['k_c']

    if k_t == 1 and k_c == 1:
        return normal_conf_pred(*params)
    elif k_t > 1 and k_c == 1:
        return conf_pred_training_poisoning(*params)
    elif k_t == 1 and k_c > 1:
        return conf_pred_calibration_poisoning(*params)
    elif k_t > 1 and k_c > 1:
        return conf_pred_both_poisoned(*params)
    else:
        raise Exception("not implemented")


def normal_conf_pred(test_idx, hparams, cal_indices, all_logits,
                     targets, hashes):
    """
        Conformal prediction under training and calibration poisoning.
    """
    k_t = hparams['k_t']
    k_c = hparams['k_c']
    alpha = hparams['alpha']
    assert k_t == 1 and k_c == 1

    logits = all_logits[0]
    idx = cal_indices[0]

    if hparams['score_function'] == "HPS":
        PS = HPS(logits, targets, alpha, idx, test_idx)
    elif hparams['score_function'] == "APS":
        PS = rep_APS(logits, targets, hashes, alpha,
                     idx, test_idx, hparams['randomized'])
    else:
        raise Exception("not implemented")

    return PS,


def conf_pred_training_poisoning(test_idx, hparams, cal_indices, all_logits,
                                 targets, hashes):
    """
        Conformal prediction under training poisoning
    """
    k_t = hparams['k_t']
    k_c = hparams['k_c']
    alpha = hparams['alpha']
    with_softmax = hparams['with_softmax']
    cal_idx = cal_indices[0]
    assert k_t > 1 and k_c == 1

    counts = np.zeros((all_logits[0].shape[0], all_logits[0].shape[1]))
    for logits in all_logits:
        counts[np.arange(counts.shape[0]), logits.argmax(axis=1)] += 1
    smoothed_logits = counts / counts.sum(axis=1)[:, None]

    PS = smoothed_HPS(smoothed_logits, targets, alpha,
                      cal_idx, test_idx, with_softmax)

    return PS, smoothed_logits


def conf_pred_calibration_poisoning(test_idx, hparams, cal_indices, all_logits,
                                    targets, hashes):
    k_t = hparams['k_t']
    k_c = hparams['k_c']
    alpha = hparams['alpha']
    assert k_t == 1 and k_c > 1

    prediction_sets = []
    for i in range(k_c):
        idx = cal_indices[i]
        logits = all_logits[0]

        if hparams['score_function'] == "HPS":
            PS = HPS(logits, targets, alpha, idx, test_idx)
        elif hparams['score_function'] == "APS":
            PS = rep_APS(logits, targets, hashes, alpha, idx,
                         test_idx, randomized=hparams['randomized'])
        else:
            raise Exception("not implemented")

        prediction_sets.append(PS)

    PS = np.array(prediction_sets).sum(axis=0)

    # build majority prediction sets
    cdfs = binom.cdf(np.arange(k_c+1), k_c, 1-alpha)
    tau = np.argmax(cdfs[cdfs <= alpha])
    PS = PS > tau

    return PS, prediction_sets, tau


def conf_pred_both_poisoned(test_idx, hparams, cal_indices, all_logits,
                            targets, hashes):
    """
        Conformal prediction under training and calibration poisoning.
    """
    k_t = hparams['k_t']
    k_c = hparams['k_c']
    alpha = hparams['alpha']
    with_softmax = hparams['with_softmax']
    assert k_t > 1 and k_c > 1

    counts = np.zeros((all_logits[0].shape[0], all_logits[0].shape[1]))
    for logits in all_logits:
        counts[np.arange(counts.shape[0]), logits.argmax(axis=1)] += 1
    smoothed_logits = counts / counts.sum(axis=1)[:, None]

    prediction_sets = []
    for i in tqdm(range(k_c)):
        cal_idx = cal_indices[i]
        PS = smoothed_HPS(smoothed_logits, targets,
                          alpha, cal_idx, test_idx, with_softmax)
        prediction_sets.append(PS)

    PS = np.array(prediction_sets).sum(axis=0)

    # build majority prediction sets
    cdfs = binom.cdf(np.arange(k_c+1), k_c, 1-alpha)
    tau = np.argmax(cdfs[cdfs <= alpha])
    PS = PS > tau

    return PS, smoothed_logits, tau

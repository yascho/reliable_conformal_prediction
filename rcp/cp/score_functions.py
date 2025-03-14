import numpy as np
import torch.nn.functional as F
from scipy.special import softmax
from scipy.stats import binom


def smoothed_HPS(smoothed_logits, y, alpha, cal_idx, val_idx, with_softmax):
    # Calibration
    n = len(cal_idx)
    if with_softmax:
        scores = softmax(smoothed_logits[cal_idx], axis=1)
    else:
        scores = smoothed_logits[cal_idx]

    scores = scores[np.arange(n), y[cal_idx]]
    q_level = np.floor((n+1)*alpha - 1)/n
    assert 0 <= q_level, f"{n} are not enough calibration points \
        (alpha={alpha}, q_level={q_level}) :("
    qhat = np.quantile(scores, q_level, method='lower')

    # Form prediction sets
    if with_softmax:
        PS = softmax(smoothed_logits[val_idx], axis=1) >= qhat
    else:
        PS = smoothed_logits[val_idx] >= qhat

    return PS


def HPS(logits, y, alpha, cal_idx, val_idx):
    """
        Homogeneous prediction sets (HPS)
        Threshold prediction sets (TPS)
        (Sadinle et al., 2019)
    """
    # Calibration
    n = len(cal_idx)
    scores = softmax(logits[cal_idx, :], axis=1)
    scores = scores[np.arange(n), y[cal_idx]]
    q_level = np.floor((n+1)*alpha - 1)/n
    assert 0 <= q_level, f"{n} are not enough calibration points \
        (alpha={alpha}, q_level={q_level}): ("
    qhat = np.quantile(scores, q_level, method='lower')

    # Form prediction sets
    prediction_sets = softmax(logits[val_idx, :], axis=1) >= qhat
    return prediction_sets


def rep_APS(logits, y, hashes, alpha, cal_idx, val_idx, randomized=False):
    """
        Reproducible APS: Adaptive prediction sets (Romano et al., 2020)
        with our reliability feature (for randomized=`reproducible`).
    """

    # Get (cumulative) scores
    n = len(cal_idx)
    scores = APS_cum_scores(logits, y, hashes, cal_idx, randomized=randomized)

    # Calibration
    q_level = np.ceil((n+1)*(1-alpha))/n
    qhat = np.quantile(scores, q_level, method='higher')

    # Form prediction sets
    smx = softmax(logits[val_idx], axis=1)
    args = smx.argsort(1)[:, ::-1]
    reordered_smx = np.take_along_axis(smx, args, axis=1)

    if randomized == 'reproducible':
        u_vec = reproducible_uniform(0, 1, hashes[val_idx], size=1)
    elif randomized == 'random':
        u_vec = reproducible_uniform(0, 1, np.random.randint(
            0, 1000, size=hashes[val_idx].shape[0]), size=1)
    else:  # not randomized APS
        u_vec = 1

    cumsums = reordered_smx.cumsum(axis=1) + (u_vec-1) * reordered_smx

    cond = cumsums <= qhat
    prediction_sets = np.take_along_axis(cond, args.argsort(axis=1), axis=1)

    return prediction_sets


def APS_cum_scores(logits, y, hashes, idx, randomized=False):
    n = len(idx)
    smx = softmax(logits[idx], axis=1)
    target_smx = smx[np.arange(n), y[idx]]

    if randomized == 'reproducible':
        u_vec = reproducible_uniform(0, 1, hashes[idx]).squeeze()
    elif randomized == 'random':
        u_vec = reproducible_uniform(0, 1, np.random.randint(
            0, 1000, size=hashes[idx].shape[0])).squeeze()
    else:
        u_vec = 1

    scores = (smx*(smx > np.expand_dims(target_smx, 1))).sum(1)
    scores += u_vec*target_smx

    return scores


def reproducible_uniform(low, high, hashes, size=1):
    if type(low) is int and type(high) is int:
        low = [low]*len(hashes)
        high = [high]*len(hashes)

    result = []
    for i in range(len(hashes)):
        rng = np.random.default_rng(seed=hashes[i])
        result.append(rng.uniform(low[i], high[i], size))
    return np.array(result)

import numpy as np
import torch.nn.functional as F
from scipy.special import softmax
from scipy.stats import binom
from tqdm.auto import tqdm
import time


def reliability_eval(hparams, targets, cal_indices, val_idx, conformal_result):
    stats_certificates = {}

    k_t = hparams['k_t']
    k_c = hparams['k_c']
    alpha = hparams['alpha']
    min_partition_size = min([len(x) for x in cal_indices])

    if k_t == 1 and k_c == 1:
        return stats_certificates
    elif k_t > 1 and k_c == 1:
        cal_idx = cal_indices[0]
        PS, smoothed_logits = conformal_result
        R, CR, SR = certify_training_poisoning(
            smoothed_logits, targets, PS, alpha, cal_idx, val_idx, k_t)
    elif k_t == 1 and k_c > 1:
        PS, prediction_sets, tau = conformal_result
        R, CR, SR = certify_calibration_poisoning(
            k_c, prediction_sets, tau, min_partition_size, alpha)
    elif k_t > 1 and k_c > 1:
        PS, smoothed_logits, tau = conformal_result
        params = smoothed_logits, PS, tau, targets, k_t, k_c, alpha, \
            cal_indices, val_idx, min_partition_size
        R, CR, SR = certify_both(*params)

    stats_certificates['robust'] = R
    stats_certificates['coverage_reliable'] = CR
    stats_certificates['size_reliable'] = SR
    return stats_certificates


def certify_calibration_poisoning(k_c, prediction_sets, tau,
                                  min_partition_size, alpha):

    robust = []
    coverage_reliable = []
    size_reliable = []

    for rho in range(1, k_c+1):
        PS_lower = np.array(prediction_sets).sum(axis=0) - rho > tau
        PS = np.array(prediction_sets).sum(axis=0) > tau
        PS_upper = np.array(prediction_sets).sum(axis=0) + rho > tau

        rob_del = min_partition_size - rho >= 1/alpha - 1

        robust.append(((PS_lower == PS_upper).all(axis=1)*rob_del).mean())
        coverage_reliable.append(((PS_lower == PS).all(axis=1)*rob_del).mean())
        size_reliable.append(((PS_upper == PS).all(axis=1)*rob_del).mean())
    robust = np.array(robust)
    coverage_reliable = np.array(coverage_reliable)
    size_reliable = np.array(size_reliable)

    return robust, coverage_reliable, size_reliable


def certify_training_poisoning(smoothed_logits, y, PS, alpha,
                               cal_idx, val_idx, k_t):
    c = y.max()+1
    n = len(cal_idx)
    q_level = np.floor((n+1)*alpha - 1)/n

    lower, upper = worst_case_smx(smoothed_logits[cal_idx], y[cal_idx], k_t, c)

    qhat_lower = []
    qhat_upper = []
    for rho in range(1, k_t+1):
        upper_scores = upper[rho-1][np.arange(n), y[cal_idx]]
        lower_scores = lower[rho-1][np.arange(n), y[cal_idx]]
        qhat_lower.append(np.quantile(lower_scores, q_level, method='lower'))
        qhat_upper.append(np.quantile(upper_scores, q_level, method='lower'))

    m = len(val_idx)
    PS_EMPTY = PS.sum(1) == 0
    PS_FULL = PS.sum(1) == c
    PS_SUM = PS.sum(1)

    inf_scores = softmax(smoothed_logits[val_idx], axis=1)

    smallest_classes_in = np.argsort(inf_scores*PS,
                                     axis=1)[:, ::-1][np.arange(m), PS_SUM-1]
    largest_classes_out = np.argsort(inf_scores*(~PS), axis=1)[:, -1]

    lower, _ = worst_case_smx(
        smoothed_logits[val_idx], smallest_classes_in, k_t, c)
    _, upper = worst_case_smx(
        smoothed_logits[val_idx], largest_classes_out, k_t, c)

    coverage_reliable = []
    size_reliable = []

    for rho in range(k_t):
        coverage_reliable.append(
            lower[rho][np.arange(m), smallest_classes_in] >= qhat_upper[rho])
        size_reliable.append(
            upper[rho][np.arange(m), largest_classes_out] < qhat_lower[rho])
        # empty sets should count as coverage-reliable
        coverage_reliable[-1][PS_EMPTY] = True
        # full sets should count as size-reliable
        size_reliable[-1][PS_FULL] = True

    robust = (np.array(coverage_reliable)*np.array(size_reliable)).mean(1)
    coverage_reliable = np.array(coverage_reliable).mean(1)
    size_reliable = np.array(size_reliable).mean(1)

    return robust, coverage_reliable, size_reliable


def certify_both(smoothed_logits, PS, tau, y, k_t, k_c, alpha,
                 cal_indices, test_idx, min_partition_size):
    c = y.max()+1

    wc_quantiles_lower = []
    wc_quantiles_upper = []

    for i in tqdm(range(k_c)):
        cal_idx = cal_indices[i]
        n = len(cal_idx)
        q_level = np.floor((n+1)*alpha - 1)/n

        lower, upper = worst_case_smx(
            smoothed_logits[cal_idx], y[cal_idx], k_t, c)

        qhat_lower = []
        qhat_upper = []
        for rho in range(1, k_t+1):
            lower_scores = lower[rho-1][np.arange(n), y[cal_idx]]
            upper_scores = upper[rho-1][np.arange(n), y[cal_idx]]

            qhat_lower.append(np.quantile(
                lower_scores, q_level, method='lower'))
            qhat_upper.append(np.quantile(
                upper_scores, q_level, method='lower'))

        wc_quantiles_lower.append(qhat_lower)
        wc_quantiles_upper.append(qhat_upper)

    wc_quantiles_lower = np.array(wc_quantiles_lower)
    wc_quantiles_upper = np.array(wc_quantiles_upper)

    # worst-case softmax changes
    lowers = []
    uppers = []
    for i in tqdm(range(c)):
        j = np.array([i]*len(test_idx))
        lower, upper = worst_case_smx(smoothed_logits[test_idx], j, k_t, c)
        lowers.append([lo[:, i] for lo in lower])
        uppers.append([up[:, i] for up in upper])
    lower = np.array(lowers)
    upper = np.array(uppers)

    samples, classes = np.where(PS)
    not_samples, not_classes = np.where(~PS)

    coverage_reliable = np.empty((k_t, k_c), dtype=object)
    size_reliable = np.empty((k_t, k_c), dtype=object)
    CR = np.zeros((k_t, k_c))
    SR = np.zeros((k_t, k_c))
    ROB = np.zeros((k_t, k_c))

    for r_t in tqdm(range(k_t)):
        for r_c in range(k_c):

            rob_del = min_partition_size - r_c >= 1/alpha - 1

            # count in how many calibration splits the coverage holds
            coverage_counts = np.zeros_like(PS, dtype=int)
            coverage_counts[samples, classes] = (
                lower[classes, r_t, samples][:, None] >=
                wc_quantiles_upper[:, r_t]).sum(1)
            coverage_reliable[r_t, r_c] = (
                (coverage_counts > tau + r_c + 1) == PS).all(1)
            CR[r_t, r_c] = (coverage_reliable[r_t, r_c]*rob_del).mean()

            # count in how many calibration splits the size is reliable
            size_counts = np.zeros_like(PS, dtype=int)
            size_counts[not_samples, not_classes] = (
                upper[not_classes, r_t, not_samples][:, None] <
                wc_quantiles_lower[:, r_t]).sum(1)
            size_reliable_per_class = (k_c-(size_counts - r_c-1) <= tau)
            size_reliable_per_class[samples, classes] = True
            size_reliable[r_t, r_c] = size_reliable_per_class.all(1)
            SR[r_t, r_c] = (size_reliable[r_t, r_c]*rob_del).mean()

            # both
            ROB[r_t, r_c] = (size_reliable[r_t, r_c] *
                             coverage_reliable[r_t, r_c]*rob_del).mean()

    return ROB, CR, SR


def worst_case_smx(smoothed_logits, targets, k_t, c):
    """
        Computes the worst-case softmax scores given training radius k_t.
    """
    n = smoothed_logits.shape[0]
    delta = 1/k_t

    cur = smoothed_logits.copy()

    wc_max_smx = []

    for _ in range(k_t):
        # add to target
        cur[np.arange(n), targets] += delta
        cur = np.minimum(cur, 1)

        # subtract from largest
        argsort = np.argsort(cur, axis=1)
        argsort = argsort[np.where(
            argsort != targets[:, None])].reshape(n, c-1)
        idx_largest = argsort[:, -1]
        cur[np.arange(n), idx_largest] -= delta
        cur = np.maximum(cur, 0)

        wc_max_smx.append(softmax(cur, axis=1))

    cur = smoothed_logits.copy()
    wc_min_smx = []

    for _ in range(k_t):
        # add to largest which is not the target
        argsort = np.argsort(cur, axis=1)
        argsort = argsort[np.where(
            argsort != targets[:, None])].reshape(n, c-1)
        idx_largest = argsort[:, -1]  # largest which is not the target
        cur[np.arange(n), idx_largest] += delta
        cur = np.minimum(cur, 1)

        # now subtract from target
        # unless target is zero, then subtract from smallest > 0

        # subtract from non-zero target
        target_zero = cur[np.arange(n), targets] == 0
        cur[~target_zero, targets[~target_zero]] -= delta

        # subtract from smallest greater zero when target is zero
        cur2 = cur.copy()
        cur2[cur == 0] = np.inf
        argsort = np.argsort(cur2, axis=1)
        smallest_gz = argsort[target_zero, 0]  # smallest > 0
        cur[target_zero, smallest_gz] -= delta
        cur = np.maximum(cur, 0)

        wc_min_smx.append(softmax(cur, axis=1))

    return wc_min_smx, wc_max_smx

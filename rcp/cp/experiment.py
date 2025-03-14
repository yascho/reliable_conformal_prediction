import numpy as np
import logging
import torch
import random
import seml
import os
import time

from rcp.datasets import *
from rcp.models import *
from rcp.utils import set_random_seed

from conformal_prediction import *
from prediction import *
from certificate import *


class Experiment():

    def run(self, hparams):
        print(hparams)
        results = {}
        dict_to_save = {}

        df = seml.get_results(hparams['seml_training_collection'],
                              to_data_frame=True)

        cond = (df['config.hparams.arch'] == hparams['arch']) & \
            (df['config.hparams.dataset'] == hparams['dataset']) & \
            (df['config.hparams.pretrained'] == hparams['pretrained']) &\
            (df['config.hparams.k_t'] == hparams['k_t'])

        df = df[cond]
        assert len(df) == 5, len(df)

        model_ids = df['_id'].to_list()
        dict_to_save['stats'] = self.experiment(model_ids, hparams)

        dict_to_save['hparams'] = hparams
        return results, dict_to_save

    def experiment(self, model_ids, hparams):
        stats = {}

        print(model_ids)
        for i, id in tqdm(enumerate(model_ids)):

            path = hparams['model_dir'] + hparams['seml_training_collection']
            path = path + "_" + str(id)
            experiment = torch.load(path)
            model_hparams = experiment['hparams']
            _, test_data, means, stds = load_dataset(model_hparams)

            # compute conformal sets for multiple calibration/test splits
            for j in range(hparams['num_cp_splits']):
                val_idx = np.arange(len(test_data))
                set_random_seed(seed=j)
                np.random.shuffle(val_idx)
                cal_idx = val_idx[:1000]
                test_idx = val_idx[1000:]

                params = (test_data, means, stds, experiment,
                          cal_idx, test_idx, hparams)
                stats[f"({i},{j})"] = calibrate_and_predict(*params)

        return stats


def calibrate_and_predict(test_data, means, stds, experiment, cal_idx,
                          test_idx, hparams):

    start_all_time = time.time()

    cal_indices = [[] for _ in range(hparams['k_c'])]
    for idx, (image, label) in tqdm(enumerate(test_data),
                                    total=len(test_data)):
        if idx not in cal_idx:
            continue
        index = int(image.sum()) % hparams['k_c']
        cal_indices[index].append(idx)
    cal_indices = [np.array(x) for x in cal_indices]

    params = (test_data, means, stds, experiment, hparams, test_idx)
    all_logits, targets, hashes, accs, stats_time = prediction(*params)

    start_cp_time = time.time()
    params = (test_idx, hparams, cal_indices, all_logits, targets, hashes)
    conformal_result = conf_pred(*params)
    end_cp_time = time.time()
    stats_time['conformal_prediction_time'] = end_cp_time - start_cp_time

    end_all_time = time.time()
    stats_time['all'] = end_all_time - start_all_time

    num_classes = experiment['hparams']['out_channels']

    # conformal prediction eval
    PS = conformal_result[0]
    PS_val = PS[np.arange(len(test_idx)), targets[test_idx]]
    stats_prediction_sets = {
        "marginal_coverage": PS_val.mean(),
        "avg_set_size": PS.sum(1).mean(),
        "singleton_hit_ratio": PS_val[PS.sum(1) == 1].mean(),
        "singleton_proportion": (PS.sum(1) == 1).sum()/PS.shape[0],
        "zero_sets": (PS.sum(1) == 0).sum()/PS.shape[0],
        "full_sets": (PS.sum(1) == num_classes).sum()/PS.shape[0]
    }

    # robustness evaluation
    stats_certificates = {}
    if hparams['score_function'] == "HPS":
        start_certificate = time.time()
        stats_certificates = reliability_eval(
            hparams, targets, cal_indices, test_idx, conformal_result)
        end_certificate = time.time()
        stats_time['certificate'] = end_certificate - start_certificate

    result = {
        "stats_prediction_sets": stats_prediction_sets,
        "stats_certificates": stats_certificates,
        "stats_time": stats_time
    }

    return result

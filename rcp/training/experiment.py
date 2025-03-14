import numpy as np
import logging
import torch
import random
import os
import time

from rcp.datasets import *
from rcp.models import *
from rcp.utils import set_random_seed
from training import *


class Experiment():

    def run(self, hparams):
        results = {}
        dict_to_save = {}

        list_train_data, test_data, means, stds = load_dataset(hparams)

        models = []
        training_times = []

        for i, train_data in tqdm(enumerate(list_train_data)):

            set_random_seed(hparams['model_seed'] * (i+1))
            model = create_image_classifier(hparams, means[i], stds[i])

            start = time.time()
            set_random_seed(hparams['model_seed'] * (i+1))
            model = train_image_classifier(model, train_data, hparams)
            end = time.time()

            models.append(model.cpu().state_dict())
            training_times.append(end - start)

        dict_to_save['models'] = models
        dict_to_save['training_time'] = training_times
        dict_to_save['hparams'] = hparams

        return results, dict_to_save

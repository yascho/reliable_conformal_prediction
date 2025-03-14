import numpy as np
import torch.nn.functional as F
from scipy.special import softmax
from scipy.stats import binom
from tqdm.auto import tqdm

import time

from rcp.models import *


def prediction(test_data, means, stds, experiment, hparams, test_idx):

    k_t = hparams['k_t']
    alpha = hparams['alpha']

    stats_time = {}
    stats_time['inference'] = [0 for _ in range(k_t)]

    model_hparams = experiment['hparams']
    batch_size = model_hparams['batch_size_inference']
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=0)
    all_logits = []
    accs = []
    for i in tqdm(range(k_t)):
        # load model
        model_state = experiment['models'][i]
        model = create_image_classifier(model_hparams, means[i], stds[i])
        model.load_state_dict(model_state)

        # full inference for model i
        start_inference = time.time()
        predictions = []
        logits = []
        targets = []
        hashes = []
        with torch.no_grad():
            for (x, y) in test_loader:
                x, y = x.to(hparams["device"]), y.to(hparams["device"])
                logits.append(model(x))
                predictions.append((logits[-1].argmax(1) == y))
                targets.append(y)
                hashes.append(x.sum((1, 2, 3)).int())
        predictions = torch.cat(predictions)
        end_inference = time.time()
        stats_time['inference'][i] = end_inference-start_inference

        accs.append(predictions[test_idx].float().mean().item())
        logits = torch.cat(logits).cpu().numpy()
        targets = torch.cat(targets).cpu().numpy()
        hashes = torch.cat(hashes).cpu().int().numpy()

        all_logits.append(logits)
    return all_logits, targets, hashes, accs, stats_time

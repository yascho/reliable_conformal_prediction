import logging
import numpy as np
import seml
import torch
from seml.experiment import Experiment as SemlExperiment
import time

from experiment import Experiment

ex = SemlExperiment()


@ex.automain
def run(_config, conf: dict, hparams: dict):

    start = time.time()
    experiment = Experiment()
    results, dict_to_save = experiment.run(hparams)
    end = time.time()
    print(f"time={end-start}s")
    results['time'] = end-start

    save_dir = conf["save_dir"]
    run_id = _config['overwrite']
    db_collection = _config['db_collection']

    if conf["save"]:
        torch.save(dict_to_save, f'{save_dir}/{db_collection}_{run_id}')
    return results

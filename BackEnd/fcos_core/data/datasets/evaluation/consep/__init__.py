import logging
import os
from datetime import datetime

import numpy as np

from .lym_eval import do_consep_evaluation

def consep_evaluation(dataset, predictions, output_folder, ovthresh):
    logger = logging.getLogger("DDTNet.inference")
    logger.info("performing consep evaluation, ignored iou_types.")
    return do_consep_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        ovthresh=ovthresh,
    )
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import logging
import os


def create_logger(logging_dir, rank, filename="log"):
    if rank == 0 and logging_dir is not None:
        os.makedirs(logging_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(logging_dir, f"{filename}.txt"), encoding="utf-8"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def get_latest_ckpt(checkpoint_dir):
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return None
    step_dirs = [
        d for d in os.listdir(checkpoint_dir)
        if d.isdigit() and os.path.isdir(os.path.join(checkpoint_dir, d))
    ]
    if not step_dirs:
        return None
    return os.path.join(checkpoint_dir, sorted(step_dirs, key=int)[-1])

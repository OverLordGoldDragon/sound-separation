import argparse
import pathlib
from typing import Dict

import torch
import tensorflow as tf  # type: ignore

from tdcnpp import TDCNpp

"""Utility to convert TensorFlow TDCN++ checkpoints to PyTorch state_dict.

This script assumes you have already exported a TF *training* checkpoint (not
an inference graph). It maps variable names according to the reference
implementation shipped with the Google-research sound-separation repo.

Usage (examples):

    python convert_tf_checkpoint.py \
        --tf_checkpoint /path/to/model.ckpt-100000 \
        --output_path tdcnpp.pt

You must install TensorFlow 1.x (`pip install tensorflow==1.15.5`) and PyTorch
(`pip install torch`).
"""

# -----------------------------------------------------------------------------
# Variable name mapping helper
# -----------------------------------------------------------------------------

def build_name_map() -> Dict[str, str]:
    """Returns dict mapping TF variable names → PyTorch parameter names."""
    mapping: Dict[str, str] = {}
    # Example entry:
    # mapping['improved_tdcn/conv_block_0/dense1/kernel'] = 'blocks.0.dense1.weight'
    # You will need to extend this list – only a few patterns are shown as
    # reference to get you started. It tends to be a straightforward regex.
    return mapping


# -----------------------------------------------------------------------------
# Conversion logic
# -----------------------------------------------------------------------------

def convert(tf_checkpoint: str, output_path: str):
    reader = tf.train.load_checkpoint(tf_checkpoint)
    tf_vars = reader.get_variable_to_shape_map()

    state_dict = {}
    name_map = build_name_map()

    for tf_name in tf_vars:
        if tf_name not in name_map:
            continue  # Skip irrelevant tensors (optimizer slots, global steps…)
        pt_name = name_map[tf_name]
        tensor = reader.get_tensor(tf_name)
        state_dict[pt_name] = torch.tensor(tensor)

    torch.save(state_dict, output_path)
    print(f"Saved PyTorch checkpoint to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf_checkpoint", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    convert(args.tf_checkpoint, args.output_path)

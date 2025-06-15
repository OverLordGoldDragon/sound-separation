import pathlib

import numpy as np
import tensorflow.compat.v1 as tf

# Disable TF2 behavior
tf.disable_eager_execution()

# -----------------------------------------------------------------------------
# Configuration - edit these paths as needed
# -----------------------------------------------------------------------------

# Path to input mixture magnitude spectrogram (B, T, F)
INPUT_PATH = r"../outs/mixture.npy"

# Path to TensorFlow checkpoint prefix (without extension)
CKPT_PATH = r"C:\Desktop\Code\sound-separation\FUSS_baseline_model\baseline_model\baseline_model"

# Output path for reference results
OUTPUT_PATH = r"../outs/reference.npy"

# -----------------------------------------------------------------------------
# Small utility that loads a TDCN++ TensorFlow-1 checkpoint, runs the network on
# a given input and stores the output to disk so that you can compare against
# the PyTorch port.
# -----------------------------------------------------------------------------

# The TF implementation lives inside models/train/ .  Add repo root to
# PYTHONPATH so `import models.train.network` works even when this script is
# executed from an arbitrary location.
THIS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
import sys
sys.path.insert(0, str(REPO_ROOT))

from models.train import network, network_config  # noqa: E402

# -----------------------------------------------------------------------------
# Patch for TF 1.15.5 compatibility: ensure LayerNormalizationScalarParams has
# attribute `_fused` expected inside its `call` method.
# -----------------------------------------------------------------------------
if not hasattr(network.LayerNormalizationScalarParams, "_fused"):
    # Add the attribute with a safe default (False -> unfused code path)
    network.LayerNormalizationScalarParams._fused = False

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_input(input_path: pathlib.Path) -> np.ndarray:
    """Loads input mixture spectrogram saved as .npy or .npz.

    Expected shape (B, T, F) – batch, frames, feature bins.
    """
    if input_path.suffix == ".npz":
        arr = np.load(input_path)["arr_0"]
    else:
        arr = np.load(input_path)
    if arr.ndim != 3:
        raise ValueError("Input must have shape (batch, frames, bins)")
    return arr.astype(np.float32)


def build_conv_tasnet_graph_matching_checkpoint(batch: int, frames: int, bins: int):
    """Build a graph that exactly matches the checkpoint variable names."""
    # The checkpoint uses conv_tasnet scope, so we need to match that
    with tf.variable_scope("conv_tasnet"):
        mixture_ph = tf.placeholder(tf.float32, shape=[batch, frames, bins], name="mixture_input")
        
        # Expand dims to match network expectation: (B, T, 1, F)
        mixture_exp = tf.expand_dims(mixture_ph, axis=2)
        
        # Build TDCN++ config (matches the baseline model)
        cfg = network_config.improved_tdcn(
            bottleneck=256,
            conv_channels=512,
            kernel_size=3,
            num_dilations=8,
            num_repeats=3,
            norm_type="global_layer_norm",
            add_skip_residual_connections=False,
            scale_type="exponential"
        )
        
        # Manually create the network structure to match checkpoint scopes
        # The checkpoint expects conv_block_X/tasnet_block/... not improved_tdcn/conv_block_X/tdcn_block/...
        
        # Initial dense layer
        initial_dense_config = network.update_config_from_kwargs(
            cfg.initial_dense_layer, num_outputs=cfg.prototype_block[0].bottleneck)
        with tf.variable_scope('initial_dense'):
            layer_activations = network.dense_layer(mixture_exp, initial_dense_config)
        
        # Process blocks directly without the improved_tdcn scope
        num_blocks = len(cfg.block_prototype_indices)
        find_scale_fn = network._find_scale_function(cfg.scale_tdcn_block)
        
        for block in range(num_blocks):
            proto_block = cfg.block_prototype_indices[block]
            scale_tdcn_block = find_scale_fn(block)
            
            with tf.variable_scope('conv_block_%d' % block):
                dilation = cfg.block_dilations[block]
                tdcn_block_config = cfg.prototype_block[proto_block]
                tdcn_block_config = network.update_config_from_kwargs(
                    tdcn_block_config,
                    scale=scale_tdcn_block,
                    dilation=dilation)
                
                # Use tasnet_block instead of tdcn_block to match checkpoint
                with tf.variable_scope('tasnet_block'):
                    dense1_config = network.update_config_from_kwargs(
                        tdcn_block_config.dense1, num_outputs=tdcn_block_config.num_conv_channels,
                        activation='linear')
                    with tf.variable_scope('dense1'):
                        y = network.dense_layer(layer_activations, dense1_config)

                    normact1_config = network.update_config_from_kwargs(
                        tdcn_block_config.normact1, activation=tdcn_block_config.middle_activation)
                    with tf.variable_scope('normact1'):
                        y = network.norm_and_activation_layer(y, normact1_config)

                    timeconv_config = network.update_config_from_kwargs(
                        tdcn_block_config.tclayer, dilation=tdcn_block_config.dilation, 
                        stride=tdcn_block_config.stride,
                        separable=tdcn_block_config.separable, 
                        kernel_size=tdcn_block_config.kernel_size)
                    with tf.variable_scope('timeconv'):
                        z = network.time_convolution_layer(y, timeconv_config)

                    normact2_config = network.update_config_from_kwargs(
                        tdcn_block_config.normact2, activation=tdcn_block_config.middle_activation)
                    with tf.variable_scope('normact2'):
                        z = network.norm_and_activation_layer(z, normact2_config)

                    dense2_config = network.update_config_from_kwargs(
                        tdcn_block_config.dense2, num_outputs=tdcn_block_config.bottleneck, 
                        scale=tdcn_block_config.scale,
                        activation=tdcn_block_config.end_of_block_activation)
                    with tf.variable_scope('dense2'):
                        z = network.dense_layer(z, dense2_config)

                    # Add residual connection from input x.
                    if tdcn_block_config.resid:
                        z = tf.add(layer_activations[..., :tdcn_block_config.bottleneck], z)
                    
                    layer_activations = z
        
        # Final reshape
        batch_size = tf.shape(layer_activations)[0]
        num_frames = tf.shape(layer_activations)[1]
        mics_and_depth = tf.shape(layer_activations)[2]
        output_size = tf.shape(layer_activations)[-1]
        
        layer_activations = tf.reshape(
            layer_activations,
            (batch_size, num_frames, mics_and_depth * output_size))
        
        return mixture_ph, layer_activations


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    inp = load_input(pathlib.Path(INPUT_PATH))
    print(f"Loaded input with shape: {inp.shape}")
    batch, frames, bins = inp.shape

    ph, out_tensor = build_conv_tasnet_graph_matching_checkpoint(batch, frames, bins)
    print(f"Built graph - output shape: {out_tensor.shape}")

    # List variables to see if they match the checkpoint
    all_vars = tf.global_variables()
    print(f"Created {len(all_vars)} variables")
    print("First few variable names:")
    for var in all_vars[:10]:
        print(f"  {var.name}")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        try:
            saver.restore(sess, CKPT_PATH)
            print("Checkpoint restored successfully")
            
            ref = sess.run(out_tensor, feed_dict={ph: inp})
            
            np.save(OUTPUT_PATH, ref)
            print(f"Saved reference output to {OUTPUT_PATH} with shape {ref.shape}")
            
        except Exception as e:
            print(f"Error restoring checkpoint: {e}")
            print("\\nCheckpoint variables (first 10):")
            ckpt_vars = tf.train.list_variables(CKPT_PATH)
            for name, shape in ckpt_vars[:10]:
                print(f"  {name}: {shape}")


if __name__ == "__main__":
    main()
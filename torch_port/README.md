# PyTorch port of TDCN++

This directory contains a minimal, idiomatic PyTorch re-implementation of the **Time-Domain Convolutional Network ++ (TDCN++)** masking network originally shipped in the TensorFlow version of *google-research/sound-separation*.

Files
-----
* `tdcnpp.py` – the network definition (`TDCNpp` & `TDCNBlock`).  
  Instantiate with `TDCNpp(in_channels=F, ...)`, where *F* is the feature dimension presented at each time step (e.g. STFT bins).
* `convert_tf_checkpoint.py` – one-off utility for converting TensorFlow checkpoints to PyTorch `state_dict`.  You **must** edit `build_name_map()` to match the variable names of your checkpoint.
* `infer.py` – ultra-simplistic demo that loads a `.pt` checkpoint, performs magnitude-only masking, and writes a separated signal.

Quick start
-----------
1. Install dependencies in your environment (PyTorch, torchaudio, soundfile, TensorFlow 1.x for conversion):

```bash
conda install pytorch torchaudio soundfile -c pytorch -c conda-forge
pip install tensorflow==1.15.5  # only needed for conversion
```

2. Convert an existing TF checkpoint:

```bash
python convert_tf_checkpoint.py --tf_checkpoint /path/to/model.ckpt-100000 --output_path tdcnpp.pt
```

3. Run inference:

```bash
python infer.py --model tdcnpp.pt --wav mixture.wav --out_dir ./outputs
```

Notes
-----
* This port keeps the same architecture but does **not** replicate training loss, STFT front-/back-end, or data I/O from the original repo.  Integrate those as needed.
* The checkpoint-conversion map is the only non-trivial part left for you to fill in once you know the variable names of your specific model.

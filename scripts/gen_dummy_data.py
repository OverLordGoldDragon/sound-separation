import numpy as np
import pathlib

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Output directory (relative to project root)
OUTPUT_DIR = pathlib.Path("../outs")

# Dummy mixture spectrogram dimensions
BATCH_SIZE = 1
TIME_FRAMES = 100  # Number of time frames
FREQ_BINS = 257    # Number of frequency bins (typical for 512-point STFT)

# Random seed for reproducibility
RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# Generate dummy data
# -----------------------------------------------------------------------------

def main():
    np.random.seed(RANDOM_SEED)
    
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate random mixture magnitude spectrogram
    # Shape: (batch, time_frames, freq_bins)
    mixture = np.random.rand(BATCH_SIZE, TIME_FRAMES, FREQ_BINS).astype(np.float32)
    
    # Scale to reasonable magnitude range (0.1 to 2.0)
    mixture = mixture * 1.9 + 0.1
    
    # Save to file
    output_path = OUTPUT_DIR / "mixture.npy"
    np.save(output_path, mixture)
    
    print(f"Generated dummy mixture data:")
    print(f"  Shape: {mixture.shape}")
    print(f"  Data type: {mixture.dtype}")
    print(f"  Min value: {mixture.min():.3f}")
    print(f"  Max value: {mixture.max():.3f}")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
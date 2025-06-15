import argparse
import pathlib
import soundfile as sf  # type: ignore
import torch
import torchaudio  # type: ignore

from tdcnpp import TDCNpp


def separate_audio(model_path: str, wav_path: str, out_dir: str):
    waveform, sr = torchaudio.load(wav_path)
    # mixture: (channels, time) – convert to mono for simple demo
    waveform = waveform.mean(dim=0, keepdim=True)

    # Compute STFT magnitude (simple demo – not ideal for production)
    stft = torch.stft(waveform, n_fft=512, hop_length=128, return_complex=True)
    magnitude = stft.abs()

    tdcn = TDCNpp(in_channels=magnitude.shape[1])
    tdcn.load_state_dict(torch.load(model_path))
    tdcn.eval()

    with torch.no_grad():
        est = tdcn(magnitude)

    # For demonstration we simply inverse-STFT the *first* estimated channel
    est_complex = est[0].unsqueeze(0) * torch.exp(1j * torch.angle(stft))
    time_signal = torch.istft(est_complex, n_fft=512, hop_length=128)

    out_path = pathlib.Path(out_dir) / "separated.wav"
    sf.write(out_path, time_signal.squeeze().cpu().numpy(), sr)
    print(f"Wrote separated audio to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pt model file")
    parser.add_argument("--wav", required=True, help="Input mixture wav")
    parser.add_argument("--out_dir", default="./outputs")
    args = parser.parse_args()

    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    separate_audio(args.model, args.wav, args.out_dir)

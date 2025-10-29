# realtime_decode_colab.py
import os, sys, argparse, yaml, re, torch, torchaudio, numpy as np
import torchaudio.compliance.kaldi as kaldi
from model.utils.init_model import init_model
from model.utils.checkpoint import load_checkpoint
from model.utils.file_utils import read_symbol_table
from model.utils.ctc_utils import get_output

# ========== setup ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _longest_suffix_prefix_overlap(a, b, max_k=32):
    k = min(max_k, len(a), len(b))
    for L in range(k, 0, -1):
        if a[-L:] == b[:L]:
            return L
    return 0

# ========== model init ==========
@torch.no_grad()
def init_model_ckpt(model_checkpoint):
    config_path = os.path.join(model_checkpoint, "config.yaml")
    checkpoint_path = os.path.join(model_checkpoint, "pytorch_model.bin")
    vocab_path = os.path.join(model_checkpoint, "vocab.txt")

    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg["encoder_conf"].pop("dynamic_chunk_sizes", None)

    model = init_model(cfg, config_path)
    load_checkpoint(model, checkpoint_path)
    model.eval().to(DEVICE)
    sym = read_symbol_table(vocab_path)
    char_dict = {v: k for k, v in sym.items()}
    return model, char_dict

# ========== streaming ==========
@torch.no_grad()
def stream_audio(args, model, char_dict):
    wav, sr = torchaudio.load(args.long_form_audio)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    wav = wav * (1 << 15)

    subsampling = model.encoder.embed.subsampling_factor
    n_layers = model.encoder.num_blocks
    conv_lorder = model.encoder.cnn_module_kernel // 2
    left = args.left_context_size
    right = args.right_context_size
    hop_samples = int(args.stream_chunk_sec * sr)
    lookahead_samples = int(args.lookahead_sec * sr)

    att_cache = torch.zeros((n_layers, left, model.encoder.attention_heads,
                             model.encoder._output_size * 2 // model.encoder.attention_heads), device=DEVICE)
    cnn_cache = torch.zeros((n_layers, model.encoder._output_size, conv_lorder), device=DEVICE)
    offset = torch.zeros(1, dtype=torch.int, device=DEVICE)

    cur = 0
    carry_text = ""
    print("Realtime decoding...")
    sys.stdout.flush()

    while cur < wav.size(1):
        seg_end = min(cur + hop_samples, wav.size(1))
        seg_end_with_look = min(seg_end + lookahead_samples, wav.size(1))
        seg = wav[:, cur:seg_end_with_look]
        if seg.size(1) < int(0.025 * sr):
            break

        x = kaldi.fbank(seg, num_mel_bins=80, frame_length=25, frame_shift=10,
                        dither=0.0, energy_floor=0.0, sample_frequency=16000).unsqueeze(0).to(DEVICE)
        x_len = torch.tensor([x.size(1)], dtype=torch.int, device=DEVICE)

        with torch.cuda.amp.autocast(enabled=(DEVICE.type=="cuda"), dtype=torch.float16):
            out, out_len, _, att_cache, cnn_cache, offset = model.encoder.forward_parallel_chunk(
                xs=x, xs_origin_lens=x_len,
                chunk_size=int((args.stream_chunk_sec / 0.01) / subsampling),
                left_context_size=left,
                right_context_size=right,
                att_cache=att_cache,
                cnn_cache=cnn_cache,
                truncated_context_size=int((args.stream_chunk_sec / 0.01) / subsampling),
                offset=offset
            )
            out = out.reshape(1, -1, out.size(-1))[:, :out_len]
            hyp = model.encoder.ctc_forward(out).squeeze(0)

        text = get_output([hyp.cpu()], char_dict)[0]
        ov = _longest_suffix_prefix_overlap(carry_text, text)
        merged = carry_text + text[ov:]
        carry_text = merged

        # in trên một dòng, ghi đè
        print("\r" + carry_text.strip(), end="")
        sys.stdout.flush()

        cur = seg_end
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

# ========== main ==========
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_checkpoint", required=True)
    p.add_argument("--long_form_audio", required=True)
    p.add_argument("--left_context_size", type=int, default=128)
    p.add_argument("--right_context_size", type=int, default=32)
    p.add_argument("--stream_chunk_sec", type=float, default=0.5)
    p.add_argument("--lookahead_sec", type=float, default=0.5)
    args = p.parse_args()

    model, char_dict = init_model_ckpt(args.model_checkpoint)
    stream_audio(args, model, char_dict)

if __name__ == "__main__":
    main()

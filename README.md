# himaTTS Hindi (hima-tts-nano-hi) as ONNX, CPU-optimized

Converted from `hima-tts-nano-hi` (62M parameters, Hindi + English words). The model card lists its license as "other": check its terms before
you redistribute these converted files. Only change to the weights: 8-bit weight quantization (see variants).

## Quick start
    pip install onnxruntime numpy sentencepiece safetensors
    python run_tts.py "यार, आज office में इतना काम था कि lunch करने का भी time नहीं मिला।" --voice Meera -o out.wav
    python run_tts.py "..." --voice Ishaan --stream           # prints time to first audio

    from hima_onnx import Hima
    h = Hima(lm="w8")                                  # recommended
    audio = h.say("नमस्ते, आप कैसे हैं?", voice="Meera", seed=0)    # float32, 24 kHz, with the original retry guard
    for chunk in h.stream("...", voice="Nikhil"): ...   # streaming
    v = h.make_random_voice(seed=3); h.say("...", voice=v)    # new random voice; save with np.savez("v.npz", **v), reuse via voice="v.npz"

Built-in voices: Meera, Sneha, Nandini (female), Siddharth, Ishaan, Nikhil (male). `voice=None` gives a random voice that stays consistent across sentences.
Text goes through the original Hindi normalizer (numbers, currency, dates, English words) in `hima_text/`.

## What is in the box
| file | size | note |
|---|---|---|
| models/lm/lm_base_w8.onnx | 65 MB | **recommended**; random voices and saved voice files (base weights) |
| models/lm/lm_voices_w8.onnx | 65 MB | **recommended** for the six built-in voices (voice adapter merged into the weights) |
| models/lm/lm_*_fp32.onnx | 228 MB each | reference precision; in the separate `hima-tts-hi-onnx-fp32-extras.zip` (unzip into the same folder, use `lm="fp32"`) |
| models/decoder/decoder_stream_fp32.onnx | 41 MB | stateful streaming Mimi decoder |
| assets/ | 18 MB | tokenizer, small embedding tables, voices, meta.json |

One graph call per frame: 12-layer backbone step (KV cache, batch of 2 for the with-text / blank-text guidance pair) + 4-step flow head + guidance mix + stop logit.
Flow time embeddings are constants and `cond_embed` runs once per frame instead of once per step.
Not included: cloning a voice from a recording (needs speechbrain ECAPA + the codec encoder). Saved voice files made by the original package work.

## Measured (one CPU core of a slow sandbox VM, 3 Hindi sentences, voice Meera; your CPU will differ)
| setup | real-time factor | first audio |
|---|---|---|
| original PyTorch fp32 | 0.68 | n/a |
| ONNX fp32 LM + streaming decoder | 0.63 | ~225 ms |
| **ONNX 8-bit LM + streaming decoder** | **0.42** | **~220 ms** |
RTF < 1 is faster than real time. Prefill (50-170 ms per sentence) is paid once per sentence. `threads=` for multi-core.

## Verified
- fp32 ONNX vs original PyTorch (temperature 0, identical inputs): identical frame counts (56/56 and 44/44). No voice: latent rel. error 1.4e-5, audio SNR 77 dB.
  Meera voice: first 50 of 56 frames match to rounding (~1e-4), then a localized 1-2% excursion in 3 late frames (float amplification), overall rel. error 2.8e-3, SNR 26 dB.
- Streaming decoder vs the original codec: max abs diff ~4e-6 for chunk sizes 1..20.
- 8-bit LM vs fp32 (teacher-forced): mean frame error 1.2-1.5% (max 5%), stop decisions agree on 100% of steps; 2.3-2.5x faster per frame.
Not verified: listening quality / CER (no audio playback or ASR here). Free-running 8-bit output differs from fp32 sample by sample (expected for autoregressive sampling).

## Important note on the codec weights
The codec checkpoint pairs rotary dimensions interleaved; a straightforward port that pairs them as halves produces a badly distorted waveform
(error ~0.37 vs the original codec). This package permutes the attention q/k rows (see `export/codec_build.py`, `export/patch_qk.py`), giving ~4e-6.
If you rebuild the decoder yourself, keep that permutation.

## Rebuild
`export/` has the export, quantization and test scripts (`export_lm_hima.py base|voices`, `export_sdec_hima.py`, `gemm2matmul.py`, `quant_hima.py`,
`parity_hima.py`, `eval_hima.py`, `benchmark.py`). They need the original model folder (paths at the top of each script) and `torch`, `transformers>=5.23`, `onnxscript`.

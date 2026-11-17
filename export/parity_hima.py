import sys, time, numpy as np, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude/hima/pkg/hima-tts-hi-onnx")
from hima_onnx import Hima, SR
lm = sys.argv[1] if len(sys.argv) > 1 else "fp32"
h = Hima("/home/claude/hima/pkg/hima-tts-hi-onnx", lm=lm, threads=1)
TEXT = "यार, आज office में इतना काम था कि lunch करने का भी time नहीं मिला।"
ids = h.sp.encode(h._normalize(TEXT, language="hi")); print("ids", len(ids), ids[:6])
for name, voice in [("Meera", "Meera"), ("novoice", None)]:
    ref_f = np.load(f"ref_{name}_frames.npy"); ref_w = np.load(f"ref_{name}_wav.npy")
    vec, ref, ad = h.resolve(voice)
    t = time.time(); fr = np.stack(list(h.frames(ids, vec, ref, ad, temperature=0.0))); tg = time.time() - t
    den = fr * h.a["emb_std"] + h.a["emb_mean"]; n = min(len(den), len(ref_f))
    t = time.time(); w = h._wav(fr); td = time.time() - t; m = min(len(w), len(ref_w))
    snr = 10 * np.log10((ref_w[:m] ** 2).sum() / (((w[:m] - ref_w[:m]) ** 2).sum() + 1e-20))
    print(f"{name:8s} frames onnx/ref {len(fr)}/{len(ref_f)} | latent max err {np.abs(den[:n]-ref_f[:n]).max():.2e} rel {np.linalg.norm(den[:n]-ref_f[:n])/np.linalg.norm(ref_f[:n]):.2e} | audio SNR {snr:.1f} dB | gen {tg:.2f}s dec {td:.2f}s RTF {(tg+td)/(len(w)/SR):.2f}")

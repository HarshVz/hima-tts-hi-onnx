import sys, time, numpy as np, onnxruntime as ort, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude/hima/pkg/hima-tts-hi-onnx")
from hima_onnx import Hima, _z, NL, H, D, DM, VOCAB
h = Hima("/home/claude/hima/pkg/hima-tts-hi-onnx", lm="fp32", threads=1)
TEXT = "यार, आज office में इतना काम था कि lunch करने का भी time नहीं मिला।"
ids = h.sp.encode(h._normalize(TEXT, language="hi"))
def sess(f):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; return ort.InferenceSession("/home/claude/hima/pkg/hima-tts-hi-onnx/models/lm/" + f, so, providers=["CPUExecutionProvider"])
def run(s, voice, adapter_file_tag):
    vec, ref, ad = h.resolve(voice)
    pc, head = h._prefix(ids, vec, ref); pu, _ = h._prefix([VOCAB] * len(ids), vec, ref)
    teacher = (np.load(f"ref_{'Meera' if voice else 'novoice'}_frames.npy") - h.a["emb_mean"]) / h.a["emb_std"]
    z = np.zeros((1, 32), np.float32); g = np.array([1.25], np.float32); out = []
    k = v = _z(NL, 2, H, 0, D); t0 = time.time()
    x, eos, k, v = s.run(None, {"embeds": np.stack([pc, pu]), "past_k": k, "past_v": v, "head": head, "noise": z, "guidance": g}); out.append((x[0], eos[0]))
    tp = time.time() - t0; t1 = time.time()
    for f in teacher:
        e = np.broadcast_to((f[None] @ h.a["frame_in"]).astype(np.float32)[None], (2, 1, DM)).copy()
        x, eos, k, v = s.run(None, {"embeds": e, "past_k": k, "past_v": v, "head": head, "noise": z, "guidance": g}); out.append((x[0], eos[0]))
    return out, tp, (time.time() - t1) / len(teacher) * 1000
for voice in ["Meera", None]:
    tag = "voices" if voice else "base"; base, tp0, ms0 = run(sess(f"lm_{tag}_fp32.onnx"), voice, tag)
    print(f"== {voice or 'no voice'}  fp32: prefill {tp0*1000:.0f} ms, {ms0:.1f} ms/frame")
    for f in [f"lm_{tag}_w8.onnx"]:
        o, tp, ms = run(sess(f), voice, tag)
        rel = [np.linalg.norm(a[0] - b[0]) / np.linalg.norm(b[0]) for a, b in zip(o, base)]; ee = [abs(a[1] - b[1]) for a, b in zip(o, base)]
        # does the stop decision (eos > -1) agree at every step?
        agree = np.mean([(a[1] > -1) == (b[1] > -1) for a, b in zip(o, base)])
        print(f"   {f:22s} frame rel err mean {np.mean(rel):.4f} max {np.max(rel):.4f} | eos err mean {np.mean(ee):.3f} max {np.max(ee):.3f} | stop-decision agreement {agree*100:.0f}% | prefill {tp*1000:.0f} ms, {ms:.1f} ms/frame")

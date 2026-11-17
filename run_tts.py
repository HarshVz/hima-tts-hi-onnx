import argparse, time, numpy as np, wave
from hima_onnx import Hima, SR
ap = argparse.ArgumentParser(description="himaTTS Hindi on ONNX Runtime")
ap.add_argument("text"); ap.add_argument("-o", "--out", default="out.wav")
ap.add_argument("--voice", default=None, help="Meera | Sneha | Nandini | Siddharth | Ishaan | Nikhil | path to a saved voice (.safetensors/.npz); omit for a random voice")
ap.add_argument("--lm", default="w8", help="w8 (default, recommended) | fp32 (needs the fp32 extras zip)")
ap.add_argument("--seed", type=int, default=None); ap.add_argument("--temp", type=float, default=0.3); ap.add_argument("--guidance", type=float, default=1.25)
ap.add_argument("--threads", type=int, default=0); ap.add_argument("--stream", action="store_true", help="stream chunks (reports time to first audio; no retry guard)")
a = ap.parse_args()
t0 = time.time(); h = Hima(lm=a.lm, threads=a.threads); h._session(h.resolve(a.voice)[2]); print(f"model load {time.time()-t0:.1f}s")
t = time.time(); first = None
if a.stream:
    ch = []
    for c in h.stream(a.text, voice=a.voice, seed=a.seed, temperature=a.temp, guidance=a.guidance):
        if first is None: first = time.time() - t
        ch.append(c)
    y = np.concatenate(ch)
else:
    y = h.say(a.text, voice=a.voice, seed=a.seed, temperature=a.temp, guidance=a.guidance)
dt = time.time() - t
with wave.open(a.out, "wb") as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
print(f"{len(y)/SR:.2f}s audio in {dt:.2f}s (RTF {dt/max(len(y)/SR,1e-9):.2f})" + (f", first audio after {first*1000:.0f} ms" if first else "") + (f", retries {h.last_retries}" if not a.stream else "") + f" -> {a.out}")

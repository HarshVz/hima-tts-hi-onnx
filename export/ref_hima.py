import sys, time, numpy as np, torch, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude/hima/clean_hi")
from hima_tts import TTS
from hima_tts.text.normalize import normalize
torch.set_grad_enabled(False)
tts = TTS("/home/claude/hima/clean_hi", device="cpu")
tts.sampler["temperature"] = 0.0
TEXT = "यार, आज office में इतना काम था कि lunch करने का भी time नहीं मिला।"
norm = normalize(TEXT, language="hi"); ids = tts._sp.encode(norm)
print("normalized:", norm); print("ids:", len(ids), ids[:12])
out = {}
for name, voice in [("Meera", "Meera"), ("novoice", None)]:
    vec, ref, adapter = tts._resolve(voice); model = tts._model(adapter)
    t = time.time()
    frames = model.generate(torch.tensor(ids), vec, ref, **tts._kw()); dt = time.time() - t
    wav = tts._codec.decode_from_latent(frames.float()[None], __import__("hima_tts.codec.stateful_module", fromlist=["x"]).init_states(tts._codec, batch_size=1, sequence_length=10000)).reshape(-1).numpy()
    print(f"{name}: {len(frames)} frames, gen {dt:.2f}s, wav {len(wav)/24000:.2f}s, peak {abs(wav).max():.3f}")
    np.save(f"ref_{name}_frames.npy", frames.numpy()); np.save(f"ref_{name}_wav.npy", wav)

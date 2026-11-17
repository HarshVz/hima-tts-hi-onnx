"""Compare speed of the ONNX variants (and, if the original hima_tts package + weights are given, PyTorch)."""
import sys, time, os, numpy as np; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hima_onnx import Hima, SR
texts = ["यार, आज office में इतना काम था कि lunch करने का भी time नहीं मिला।",
         "मैं आज बाज़ार गया था और वहाँ से सब्ज़ियाँ ख़रीद कर घर वापस आया, फिर मैंने खाना बनाया।",
         "इस किताब की कीमत ₹2,500 है। यह 15/08/2026 को प्रकाशित हुई थी। जिन्होंने इसे पढ़ा, सबको बहुत पसंद आई।"]
VOICE = "Meera"; threads = int(os.environ.get("THREADS", "1"))
print(f"{'config':34s}{'RTF':>7s}{'first audio':>13s}")
if os.environ.get("TORCH_REF"):
    sys.path.insert(0, os.environ["TORCH_REF"]); import torch; from hima_tts import TTS
    torch.set_num_threads(threads); t = TTS(os.environ["TORCH_REF"], device="cpu"); t.say(texts[0], voice=VOICE, seed=0, retries=0)
    tt = aud = 0
    for x in texts:
        t0 = time.time(); y = t.say(x, voice=VOICE, seed=0, retries=0); tt += time.time() - t0; aud += len(y) / SR
    print(f"{'PyTorch fp32 (original)':34s}{tt/aud:7.2f}{'n/a':>13s}")
for lm in ["fp32", "w8"]:
    h = Hima(lm=lm, threads=threads); h.say(texts[0], voice=VOICE, seed=0, retries=0)
    tt = aud = 0; firsts = []
    for x in texts:
        t0 = time.time(); n = 0; first = None
        for c in h.stream(x, voice=VOICE, seed=0):
            if first is None: first = time.time() - t0
            n += len(c)
        tt += time.time() - t0; aud += n / SR; firsts.append(first)
    print(f"{'ONNX LM ' + lm + ' + streaming decoder':34s}{tt/aud:7.2f}{np.mean(firsts)*1000:10.0f} ms")

"""himaTTS (Telugu / Hindi, read from assets/meta.json) on ONNX Runtime, CPU only.   pip install onnxruntime numpy sentencepiece"""
import math, os, re
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
SR, FRAME = 24000, 1920
NL, H, D, DM = 12, 9, 64, 576
VOCAB = 16000
LM = {"fp32": "lm_{v}_fp32.onnx", "w8": "lm_{v}_w8.onnx"}
DEC = {"stream": "decoder_stream_fp32.onnx"}

def _z(*s): return np.zeros(s, np.float32)
def _init_state():
    return dict(up_carry=_z(1, 512, 16), kv_k=_z(2, 1, 8, 0, 64), kv_v=_z(2, 1, 8, 0, 64), pos=np.array([0], np.int64),
                c0=_z(1, 512, 6), t1=_z(1, 256, 6), r1=_z(1, 256, 2), t2=_z(1, 128, 5), r2=_z(1, 128, 2),
                t3=_z(1, 64, 4), r3=_z(1, 64, 2), cl=_z(1, 64, 2))
_SN = list(_init_state())

def _frame_limit(norm):
    letters = sum(c.isalnum() or 0x0900 <= ord(c) <= 0x0D7F for c in norm)
    return int(np.ceil((2.0 * 0.092 * letters + 1.0) * 12.5))

def _longest_silence(a, sr=SR):
    a = np.asarray(a, np.float32).reshape(-1); w = sr // 10; n = len(a) // w
    if n == 0: return 0.0
    db = 20 * np.log10(np.sqrt((a[: n * w].reshape(n, w) ** 2).mean(1)) + 1e-9); loud = np.where(db > -55)[0]
    if len(loud) == 0: return n / 10
    run = best = 0
    for v in db[loud[0]:]:
        run = run + 1 if v <= -55 else 0; best = max(best, run)
    return best / 10

def _sentences(text, limit=220):
    parts = [p.strip() for p in re.split(r"(?<=[.!?।॥])\s+", text.strip()) if p.strip()]; out = []
    for p in parts:
        if out and len(out[-1]) + len(p) + 1 <= limit: out[-1] += " " + p
        else: out.append(p)
    return out

class Hima:
    def __init__(self, root=HERE, lm="w8", decoder="stream", threads=0, language=None):
        import json
        self.meta = json.load(open(os.path.join(root, "assets/meta.json")))
        self.root, self.lm_kind, self.language = root, lm, language or self.meta["language"]
        so = ort.SessionOptions()
        if threads: so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._so = so; self._lm = {}
        self.dec = ort.InferenceSession(os.path.join(root, "models/decoder", DEC.get(decoder, decoder)), so, providers=["CPUExecutionProvider"])
        a = np.load(os.path.join(root, "assets/assets.npz")); self.a = {k: a[k] for k in a.files}
        v = np.load(os.path.join(root, "assets/voices.npz")); self.voices = [str(n) for n in v["names"]]
        self._v = {k: v[k] for k in v.files if k != "names"}
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=os.path.join(root, "assets/tokenizer.model"))
        from hima_text.normalize import normalize
        self._normalize = normalize
        self.last_retries = 0

    def _session(self, adapter):
        key = "voices" if adapter else "base"
        if key not in self._lm:
            f = LM.get(self.lm_kind, self.lm_kind).format(v=key)
            self._lm[key] = ort.InferenceSession(os.path.join(self.root, "models/lm", f), self._so, providers=["CPUExecutionProvider"])
        return self._lm[key]

    # ---- the parts the graph does not contain (tiny matmuls) ----
    def _prefix(self, ids, vec, ref):
        a = self.a
        t = a["text_embed"][np.asarray(ids)].astype(np.float32) @ a["text_up"]
        if vec is not None:
            row = vec.astype(np.float32) @ a["spk_in_w"] + a["spk_in_b"]; head = vec.astype(np.float32) @ a["head_voice_w"]
        else: row, head = a["no_voice"], a["head_no_voice"]
        parts = [a["bos_text"][None], t, row[None]]
        if ref is not None and len(ref) > 0:
            r = (np.asarray(ref, np.float32)[:62] - a["emb_mean"]) / a["emb_std"]
            parts.append(r @ a["ref_in"] + a["ref_tag"])
        parts.append(a["bos_frame_emb"][None])
        return np.concatenate(parts, 0).astype(np.float32), head[None].astype(np.float32)

    def frames(self, ids, vec=None, ref=None, adapter=False, guidance=1.25, max_frames=375, temperature=0.3, rng=None):
        """Generator of normalised frames [32]."""
        rng = rng or np.random.default_rng(); s = self._session(adapter); sc = math.sqrt(temperature)
        pc, head = self._prefix(ids, vec, ref); pu, _ = self._prefix([VOCAB] * len(ids), vec, ref)
        g = np.array([guidance], np.float32); nz = lambda: (rng.standard_normal((1, 32)) * sc).astype(np.float32)
        k = v = _z(NL, 2, H, 0, D)
        x, eos, k, v = s.run(None, {"embeds": np.stack([pc, pu]), "past_k": k, "past_v": v, "head": head, "noise": nz(), "guidance": g})
        for t in range(max_frames):
            if float(eos[0]) > -1.0: return
            yield x[0]
            if t == max_frames - 1: return
            e = (x @ self.a["frame_in"]).astype(np.float32); e = np.broadcast_to(e[None], (2, 1, DM)).copy()
            x, eos, k, v = s.run(None, {"embeds": e, "past_k": k, "past_v": v, "head": head, "noise": nz(), "guidance": g})

    def decode_stream(self, frame_iter, first_chunk=2, chunk=4):
        st = _init_state(); buf = []; want = first_chunk
        def dec(fr):
            nonlocal st
            r = self.dec.run(None, {"lat": np.stack(fr)[None].astype(np.float32), **st}); st = dict(zip(_SN, r[1:])); return r[0][0, 0]
        for f in frame_iter:
            buf.append(f)
            if len(buf) >= want: yield dec(buf); buf = []; want = chunk
        if buf: yield dec(buf)

    # ---- voices ----
    def resolve(self, voice):
        """-> (vec, ref, adapter). voice: built-in name, a dict {"vec","ref"}, a saved .safetensors/.npz voice, or None (random voice)."""
        if voice is None: return None, None, False
        if isinstance(voice, dict): return voice.get("vec"), voice.get("ref"), False
        if voice in self.voices: return self._v[f"vec_{voice}"], self._v[f"ref_{voice}"], True
        if isinstance(voice, str) and os.path.exists(voice):
            if voice.endswith(".npz"): z = np.load(voice); return (z["vec"] if "vec" in z.files else None), z["ref"], False
            from safetensors.numpy import load_file
            z = load_file(voice); return z.get("vec"), z.get("ref"), False
        raise ValueError(f"unknown voice {voice!r}; use one of {self.voices}, a saved voice file, or None")

    def make_random_voice(self, seed=0, temperature=None):
        """New random voice (same recipe as the original: first 62 frames of a fixed sentence read by the base model)."""
        line = self.meta["random_voice_line"]
        ids = self.sp.encode(self._normalize(line, language=self.language))
        fr = np.stack(list(self.frames(ids, None, None, False, temperature=temperature or 0.3, rng=np.random.default_rng(seed))))
        return {"ref": (fr[:62] * self.a["emb_std"] + self.a["emb_mean"]).astype(np.float32)}

    def _wav(self, frames):
        if not len(frames): return np.zeros(0, np.float32)
        return np.concatenate(list(self.decode_stream(iter(frames), first_chunk=8, chunk=16)))

    def say(self, text, voice=None, seed=None, pause=0.15, retries=2, guidance=1.25, temperature=0.3, max_frames=375):
        """Whole-text synthesis like the original (with its retry guard). Returns float32 audio at 24 kHz."""
        rng = np.random.default_rng(seed); vec, ref, ad = self.resolve(voice)
        gap = np.zeros(int(pause * SR), np.float32); out = []; self.last_retries = 0
        for part in _sentences(text):
            norm = self._normalize(part, language=self.language); ids = self.sp.encode(norm)
            limit = min(max_frames, _frame_limit(norm)); best = None
            for _ in range(retries + 1):
                fr = list(self.frames(ids, vec, ref, ad, guidance, max_frames, temperature, rng)); wav = self._wav(fr)
                sil = _longest_silence(wav); bad = bool(retries) and (len(fr) >= limit or sil >= 3.0)
                if best is None or not bad or (best[3] and sil < best[2]): best = (fr, wav, sil, bad)
                if not bad: break
                self.last_retries += 1
            fr, wav = best[0], best[1]
            if voice is None and ref is None and len(fr) >= 12:
                ref = (np.stack(fr[:62]) * self.a["emb_std"] + self.a["emb_mean"]).astype(np.float32)   # keep the random voice
            if len(fr): out += [wav, gap]
        audio = np.concatenate(out[:-1]) if out else np.zeros(0, np.float32)
        pk = float(np.abs(audio).max()) if len(audio) else 0.0
        return audio * (0.97 / pk) if pk > 0.97 else audio

    def stream(self, text, voice=None, seed=None, pause=0.15, guidance=1.25, temperature=0.3, max_frames=375, first_chunk=2, chunk=4):
        """Yield audio chunks as soon as they are decoded (no retry guard). Random voices stay consistent across sentences."""
        rng = np.random.default_rng(seed); vec, ref, ad = self.resolve(voice); gap = np.zeros(int(pause * SR), np.float32); first = True
        for part in _sentences(text):
            ids = self.sp.encode(self._normalize(part, language=self.language)); got = []
            def it():
                for f in self.frames(ids, vec, ref, ad, guidance, max_frames, temperature, rng): got.append(f); yield f
            if not first: yield gap
            first = False
            yield from self.decode_stream(it(), first_chunk, chunk)
            if voice is None and ref is None and len(got) >= 12: ref = (np.stack(got[:62]) * self.a["emb_std"] + self.a["emb_mean"]).astype(np.float32)

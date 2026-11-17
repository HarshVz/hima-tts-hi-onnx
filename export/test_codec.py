import sys, numpy as np, torch, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude/hima/clean_hi")
from codec_build import build_mimi
from stream_dec import StreamDec, init_state, NAMES
torch.set_grad_enabled(False)
a = np.load("onnx/assets.npz"); mean, std = torch.from_numpy(a["emb_mean"]), torch.from_numpy(a["emb_std"])
frames = torch.from_numpy(np.load("ref_Meera_frames.npy")); ref = np.load("ref_Meera_wav.npy")
for act in ["gelu_pytorch_tanh", "gelu"]:
    m = build_mimi(act=act)
    sd = StreamDec(m, mean, std).eval(); st = {k: torch.from_numpy(v) for k, v in init_state().items()}
    outs = []; i = 0; xn = (frames - mean) / std
    for c in [1, 3, 2, 8, 5, 1, 12, 20, 40]:
        if i >= len(xn): break
        c = min(c, len(xn) - i); r = sd(xn[None, i:i + c], *[st[k] for k in NAMES]); outs.append(r[0][0, 0].numpy()); st = dict(zip(NAMES, r[1:])); i += c
    y = np.concatenate(outs); n = min(len(y), len(ref))
    print(act, "len", len(y), len(ref), "max abs err vs hima codec:", float(np.abs(y[:n] - ref[:n]).max()))

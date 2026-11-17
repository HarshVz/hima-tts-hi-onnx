import sys, numpy as np, torch, warnings, time; warnings.filterwarnings("ignore")
from codec_build import build_mimi
from stream_dec import StreamDec, init_state, NAMES, WIN
torch.set_grad_enabled(False)
a = np.load("onnx/assets.npz"); mean, std = torch.from_numpy(a["emb_mean"]), torch.from_numpy(a["emb_std"])
PERM = (sys.argv[1] != "buggy") if len(sys.argv) > 1 else True
sd = StreamDec(build_mimi(act="gelu_pytorch_tanh", permute=PERM), mean, std).eval()
st = {k: torch.from_numpy(v) for k, v in init_state().items()}
st["kv_k"] = torch.randn(2, 1, 8, 40, 64) * .1; st["kv_v"] = torch.randn(2, 1, 8, 40, 64) * .1; st["pos"] = torch.tensor([9], dtype=torch.int64)
ex = (torch.randn(1, 3, 32),) + tuple(st[k] for k in NAMES)
A = torch.export.Dim.AUTO; dyn = {"lat": {1: A}, "kv_k": {3: A}, "kv_v": {3: A}}
for k in NAMES:
    if k not in ("kv_k", "kv_v"): dyn[k] = None
ep = torch.export.export(sd, ex, dynamic_shapes=dyn, strict=False)
prog = torch.onnx.export(ep, dynamo=True, opset_version=18)
for i, n in zip(prog.model.graph.inputs, ["lat"] + NAMES): i.name = n
for o, n in zip(prog.model.graph.outputs, ["audio"] + [n + "_out" for n in NAMES]): o.name = n
prog.save("onnx/decoder_stream_fp32.onnx" if PERM else "onnx/decoder_stream_BUGGY.onnx"); print("exported")

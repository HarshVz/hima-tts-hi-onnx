"""Fix a Mimi-decoder ONNX that was built from raw (interleaved-RoPE) weights: permute q/k output features per head
so the half-split rotary used in the graph pairs the right dimensions.  usage: patch_qk.py in.onnx out.onnx"""
import sys, onnx, numpy as np
from onnx import numpy_helper
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src); g = m.graph; ini = {i.name: i for i in g.initializer}
cons = {}
for n in g.node:
    for i in n.input: cons.setdefault(i, []).append(n)
TRIV = {"Reshape", "Transpose", "Cast", "Unsqueeze", "Squeeze", "Identity"}
def reaches(out, target, depth=6):
    fr, seen = [out], set()
    for _ in range(depth):
        nxt = []
        for t in fr:
            for c in cons.get(t, []):
                if c.op_type == target: return True
                if c.op_type in TRIV and c.output[0] not in seen: seen.add(c.output[0]); nxt.append(c.output[0])
        fr = nxt
    return False
perm = np.concatenate([np.arange(0, 64, 2), np.arange(1, 64, 2)]); done = set(); n_fix = 0
for n in g.node:
    if n.op_type == "MatMul" and n.input[1] in ini and tuple(ini[n.input[1]].dims) == (512, 512) and reaches(n.output[0], "Slice") and not reaches(n.output[0], "Concat"):
        name = n.input[1]
        assert name not in done and len(cons[name]) == 1, "weight shared"
        w = numpy_helper.to_array(ini[name]); w2 = np.ascontiguousarray(w.reshape(512, 8, 64)[:, :, perm].reshape(512, 512))
        ini[name].CopyFrom(numpy_helper.from_array(w2, name)); done.add(name); n_fix += 1
print("patched q/k projections:", n_fix)
assert n_fix == 4, "expected 2 layers x (q,k)"
onnx.save(m, dst)

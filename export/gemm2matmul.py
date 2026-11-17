import onnx, numpy as np, sys
from onnx import numpy_helper, helper
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src); g = m.graph
inits = {i.name: i for i in g.initializer}
new_nodes, n_conv, made = [], 0, {}
for n in g.node:
    if n.op_type != "Gemm": new_nodes.append(n); continue
    at = {a.name: helper.get_attribute_value(a) for a in n.attribute}
    assert at.get("alpha", 1.0) == 1.0 and at.get("beta", 1.0) == 1.0 and not at.get("transA", 0)
    A, B = n.input[0], n.input[1]; C = n.input[2] if len(n.input) > 2 else None
    if B in inits:
        w = numpy_helper.to_array(inits[B])
        if at.get("transB", 0): w = w.T
        nb = B + "_mm"
        if nb not in made: g.initializer.append(numpy_helper.from_array(np.ascontiguousarray(w), nb)); made[nb] = 1
        B = nb
    else:
        if at.get("transB", 0):
            tn = B + "_T"; new_nodes.append(helper.make_node("Transpose", [B], [tn], perm=[1, 0])); B = tn
    mm_out = n.output[0] if C is None else n.output[0] + "_mm"
    new_nodes.append(helper.make_node("MatMul", [A, B], [mm_out], name=(n.name or n.output[0]) + "_mm"))
    if C is not None: new_nodes.append(helper.make_node("Add", [mm_out, C], [n.output[0]], name=(n.name or n.output[0]) + "_bias"))
    n_conv += 1
del g.node[:]; g.node.extend(new_nodes)
used = {i for n in g.node for i in n.input}
keep = [i for i in g.initializer if i.name in used]; del g.initializer[:]; g.initializer.extend(keep)
onnx.checker.check_model(m); onnx.save(m, dst); print("converted", n_conv, "Gemm ->", dst)

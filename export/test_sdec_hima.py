import numpy as np, onnxruntime as ort, time, sys
from stream_dec import init_state, NAMES
f = sys.argv[1] if len(sys.argv) > 1 else "onnx/decoder_stream_fp32.onnx"
so = ort.SessionOptions(); so.intra_op_num_threads = 1
s = ort.InferenceSession(f, so, providers=["CPUExecutionProvider"])
a = np.load("onnx/assets.npz"); fr = np.load("ref_Meera_frames.npy"); ref = np.load("ref_Meera_wav.npy")
x = ((fr - a["emb_mean"]) / a["emb_std"]).astype(np.float32)[None]
for sched in ([1,3,2,8,5,1,1,12,4,7,7,20], [4]*20, [1]*66):
    st = init_state(); outs = []; i = 0; t = time.time()
    for m in sched:
        if i >= x.shape[1]: break
        m = min(m, x.shape[1]-i); r = s.run(None, {"lat": x[:, i:i+m], **st}); outs.append(r[0][0,0]); st = dict(zip(NAMES, r[1:])); i += m
    y = np.concatenate(outs); dt = time.time()-t
    print(f"chunks~{sched[:4]} err vs hima codec {np.abs(y-ref[:len(y)]).max():.2e}  RTF {dt/(len(y)/24000):.3f}")

import sys, numpy as np, torch, yaml, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude/hima/clean_hi")
from safetensors.torch import load_file
sd = load_file("/home/claude/hima/clean_hi/model.safetensors")
g = lambda k: sd[k].float().numpy()
np.savez("onnx/assets.npz",
    text_embed=g("text_embed.weight"), text_up=g("text_up.weight").T.copy(),            # [16001,256], [256,576]
    spk_in_w=g("spk_in.weight").T.copy(), spk_in_b=g("spk_in.bias"), no_voice=g("no_voice"),
    head_voice_w=g("head_voice.weight").T.copy(), head_no_voice=g("head_no_voice"),
    bos_text=g("bos_text"), bos_frame_emb=(sd["bos_frame"].float() @ sd["frame_in.weight"].float().T).numpy(),
    frame_in=g("frame_in.weight").T.copy(), ref_in=g("ref_in.weight").T.copy(), ref_tag=g("ref_tag"),
    emb_mean=g("emb_mean"), emb_std=g("emb_std"))
v = load_file("/home/claude/hima/clean_hi/voices/voices.safetensors"); names = [x["name"] for x in json.load(open("/home/claude/hima/clean_hi/voices/voices.json"))["voices"]]
np.savez("onnx/voices.npz", names=np.array(names), **{k: x.float().numpy() for k, x in v.items()})
print({k: x.shape for k, x in np.load("onnx/assets.npz").items()}); print(names, [k for k in v][:4])
print("emb_std range", g("emb_std").min(), g("emb_std").max())

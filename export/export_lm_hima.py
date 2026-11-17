import sys, time, numpy as np, torch, torch.nn as nn, warnings, yaml
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude/hima/clean_hi")
from safetensors.torch import load_file
from hima_tts.model import ModelConfig, SpeechModel
torch.set_grad_enabled(False)
F = "/home/claude/hima/clean_hi/"
cfgd = yaml.safe_load(open(F + "config.yaml")); cfg = ModelConfig(**{k: v for k, v in cfgd["model"].items() if k in ModelConfig.__dataclass_fields__})
NL, H, D = cfg.n_begin + cfg.n_middle * cfg.n_loops + cfg.n_end, cfg.n_heads, cfg.d_model // cfg.n_heads
assert cfg.n_loops == 1

def load(adapter):
    m = SpeechModel(cfg); sd = load_file(F + "model.safetensors")
    if adapter:
        ad = load_file(F + "voices/adapter.safetensors")
        for n in {k[:-2] for k in ad if k.endswith(".A")}:
            w = n + ".weight"; sd[w] = sd[w] + (ad[n + ".B"].float() @ ad[n + ".A"].float()).to(sd[w].dtype)
    m.load_state_dict(sd, strict=True); return m.eval()

class LMStep(nn.Module):
    """backbone step (batch 2: with text / blank text) + 4-step flow head + guidance, fused."""
    def __init__(s, m):
        super().__init__()
        bb = m.backbone
        s.layers = nn.ModuleList(list(bb.begin) + list(bb.middle) + list(bb.end)); s.out_norm = bb.out_norm
        s.eos, s.flow = m.out_eos, m.flow_net
        s.register_buffer("inv_freq", 1.0 / (cfg.rope_theta ** (torch.arange(0, D, 2).float() / D)))
        te = m.flow_net.time_embed; tcs = []
        for i in range(4):
            sv, tv = torch.full((1, 1), i / 4), torch.full((1, 1), (i + 1) / 4)
            tcs.append((te[0](sv) + te[1](tv)) / 2)               # constant per step
        s.register_buffer("tc", torch.cat(tcs, 0))                 # [4,512]
    @staticmethod
    def rope(x, cos, sin):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    def forward(s, embeds, past_k, past_v, head, noise, guidance):
        B, T = embeds.shape[0], embeds.shape[1]; P = past_k.shape[3]
        pos = torch.arange(T, dtype=torch.float32) + P
        fr = pos[:, None] * s.inv_freq[None]; cos, sin = fr.cos()[None, None], fr.sin()[None, None]
        qpos = (torch.arange(T) + P)[:, None]; kpos = torch.arange(P + T)[None, :]
        mask = torch.where(kpos > qpos, torch.tensor(-1e9), torch.tensor(0.0))[None, None]
        x = embeds; nk, nv = [], []
        for i, L in enumerate(s.layers):
            h = L.norm1(x)
            q = L.qn(L.q(h).view(B, T, H, D)).transpose(1, 2)
            k = L.kn(L.k(h).view(B, T, H, D)).transpose(1, 2)
            v = L.v(h).view(B, T, H, D).transpose(1, 2)
            q, k = s.rope(q, cos, sin), s.rope(k, cos, sin)
            k = torch.cat([past_k[i], k], 2); v = torch.cat([past_v[i], v], 2); nk.append(k); nv.append(v)
            w = torch.softmax(torch.matmul(q, k.transpose(2, 3)) * (D ** -0.5) + mask, -1)
            o = L.o(torch.matmul(w, v).transpose(1, 2).reshape(B, T, H * D))
            x = x + L.ls1 * o
            x = x + L.ls2 * L.ff2(torch.nn.functional.gelu(L.ff1(L.norm2(x)), approximate="tanh"))
        z = s.out_norm(x)[:, -1]                                   # [2,576]
        eos = s.eos(z[:1])[:, 0]                                   # [1]  (row 0 = with text)
        fl = s.flow; y0 = fl.cond_embed(z + head); cur = noise.expand(2, -1)
        for i in range(4):
            y = y0 + s.tc[i]; hh = fl.input_proj(cur)
            for blk in fl.res_blocks: hh = blk(hh, y)
            cur = cur + fl.final_layer(hh, y) * 0.25
        xc, xu = cur[:1], cur[1:]
        return xu + guidance * (xc - xu), eos, torch.stack(nk), torch.stack(nv)

if __name__ == "__main__":
    which = sys.argv[1]                                           # base | voices
    m = LMStep(load(which == "voices")).eval()
    T, P = 5, 7
    ex = (torch.randn(2, T, 576), torch.randn(NL, 2, H, P, D) * .1, torch.randn(NL, 2, H, P, D) * .1, torch.randn(1, 576) * .1, torch.randn(1, 32), torch.tensor([1.25]))
    Td = torch.export.Dim("T", min=1, max=1024); Pd = torch.export.Dim("P", min=0, max=1024)
    t = time.time()
    prog = torch.onnx.export(m, ex, dynamic_shapes={"embeds": {1: Td}, "past_k": {3: Pd}, "past_v": {3: Pd}, "head": None, "noise": None, "guidance": None},
                             dynamo=True, input_names=["embeds", "past_k", "past_v", "head", "noise", "guidance"],
                             output_names=["frame", "eos", "new_k", "new_v"], opset_version=18)
    prog.save(f"onnx/lm_{which}_fp32.onnx"); print("exported", which, round(time.time() - t, 1), "s")

import torch, numpy as np, warnings, time
import torch.nn as nn, torch.nn.functional as F
warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)
WIN = 250

class StreamDec(nn.Module):
    def __init__(s, mimi, mean, std):
        super().__init__()
        d = mimi.decoder; s.lp = mimi.latent_proj; s.up = d.upsample.conv
        s.tl = d.decoder_transformer.layers; s.sn = d.decoder_seanet.layers
        s.register_buffer("mean", mean.clone()); s.register_buffer("std", std.clone())
        s.register_buffer("inv_freq", d.decoder_transformer.rotary_emb.inv_freq.clone().float())
        s.H, s.D = 8, 64
    @staticmethod
    def rot(x):
        h = x.shape[-1] // 2
        return torch.cat((-x[..., h:], x[..., :h]), -1)
    @staticmethod
    def cconv(x, conv, st, dil=1):
        k = conv.kernel_size[0]; pad = (k - 1) * dil
        if pad == 0: return conv(x), st
        xx = torch.cat([st, x], -1)
        return F.conv1d(xx, conv.weight, conv.bias, dilation=dil), xx[..., -pad:]
    @staticmethod
    def ctr(x, conv, carry, stride):
        k = conv.kernel_size[0]; ov = k - stride; L = x.shape[-1] * stride
        z = F.conv_transpose1d(x, conv.weight, None, stride=stride, groups=conv.groups)
        z = torch.cat([z[..., :ov] + carry, z[..., ov:]], -1)
        y = z[..., :L]
        if conv.bias is not None: y = y + conv.bias[None, :, None]
        return y, z[..., L:]
    def forward(s, lat, up_carry, kv_k, kv_v, pos, c0, t1, r1, t2, r2, t3, r3, cl):
        M = lat.shape[1]
        x = s.lp((lat * s.std + s.mean).transpose(1, 2))                        # [1,512,M]
        z = F.conv_transpose1d(x, s.up.weight, None, stride=16, groups=512)
        z = torch.cat([z[..., :16] + up_carry, z[..., 16:]], -1)
        y = z[..., :16 * M]; n_up = z[..., 16 * M:]
        h = y.transpose(1, 2); T = 16 * M; Pk = kv_k.shape[3]
        qp = pos[0] + torch.arange(T); kp = (pos[0] - Pk) + torch.arange(Pk + T)
        ok = (kp[None, :] <= qp[:, None]) & (kp[None, :] > qp[:, None] - WIN) & (kp[None, :] >= 0)
        mask = torch.where(ok, torch.tensor(0.0), torch.tensor(-1e9))[None, None]
        fr = qp.float()[:, None] * s.inv_freq[None]; emb = torch.cat((fr, fr), -1)
        cos, sin = emb.cos()[None, None], emb.sin()[None, None]
        nks, nvs = [], []
        for i, ly in enumerate(s.tl):
            a = ly.self_attn; r = h; hn = ly.input_layernorm(h)
            q = a.q_proj(hn).view(1, T, s.H, s.D).transpose(1, 2)
            k = a.k_proj(hn).view(1, T, s.H, s.D).transpose(1, 2)
            v = a.v_proj(hn).view(1, T, s.H, s.D).transpose(1, 2)
            q = q * cos + s.rot(q) * sin; k = k * cos + s.rot(k) * sin
            k = torch.cat([kv_k[i], k], 2); v = torch.cat([kv_v[i], v], 2)
            nks.append(k[:, :, -WIN:]); nvs.append(v[:, :, -WIN:])
            w = torch.softmax(torch.matmul(q, k.transpose(2, 3)) * (s.D ** -0.5) + mask, -1)
            o = a.o_proj(torch.matmul(w, v).transpose(1, 2).reshape(1, T, s.H * s.D))
            h = r + ly.self_attn_layer_scale(o)
            h = h + ly.mlp_layer_scale(ly.mlp(ly.post_attention_layernorm(h)))
        a = h.transpose(1, 2)                                                   # [1,512,T]
        sn = s.sn
        a, n_c0 = s.cconv(a, sn[0].conv, c0)
        a, n_t1 = s.ctr(F.elu(a), sn[2].conv, t1, 6)
        rb = sn[3]; b, n_r1 = s.cconv(F.elu(a), rb.block[1].conv, r1); a = a + rb.block[3].conv(F.elu(b))
        a, n_t2 = s.ctr(F.elu(a), sn[5].conv, t2, 5)
        rb = sn[6]; b, n_r2 = s.cconv(F.elu(a), rb.block[1].conv, r2); a = a + rb.block[3].conv(F.elu(b))
        a, n_t3 = s.ctr(F.elu(a), sn[8].conv, t3, 4)
        rb = sn[9]; b, n_r3 = s.cconv(F.elu(a), rb.block[1].conv, r3); a = a + rb.block[3].conv(F.elu(b))
        a, n_cl = s.cconv(F.elu(a), sn[11].conv, cl)
        return (a, n_up, torch.stack(nks), torch.stack(nvs), pos + T, n_c0, n_t1, n_r1, n_t2, n_r2, n_t3, n_r3, n_cl)

def init_state():
    z = lambda *sh: np.zeros(sh, np.float32)
    return dict(up_carry=z(1, 512, 16), kv_k=z(2, 1, 8, 0, 64), kv_v=z(2, 1, 8, 0, 64), pos=np.array([0], np.int64),
                c0=z(1, 512, 6), t1=z(1, 512 // 2, 6), r1=z(1, 256, 2), t2=z(1, 128, 5), r2=z(1, 128, 2),
                t3=z(1, 64, 4), r3=z(1, 64, 2), cl=z(1, 64, 2))
NAMES = list(init_state().keys())

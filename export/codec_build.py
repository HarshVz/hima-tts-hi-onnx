import sys, numpy as np, torch, warnings; warnings.filterwarnings("ignore")
from safetensors.torch import load_file
from transformers.models.pocket_tts_audio_tokenizer.configuration_pocket_tts_audio_tokenizer import PocketTTSAudioTokenizerConfig
from transformers.models.pocket_tts_audio_tokenizer.modeling_pocket_tts_audio_tokenizer import PocketTTSAudioTokenizerModel
def build_mimi(path="/home/claude/hima/clean_hi/codec.safetensors", act="gelu_pytorch_tanh", permute=True):
    raw = {k[5:]: v for k, v in load_file(path).items() if k.startswith("mimi.")}
    sd = {}
    for k, v in raw.items():
        if k.startswith("decoder.model."):
            sd[k.replace("decoder.model.", "decoder.decoder_seanet.layers.").replace(".convtr.", ".conv.")] = v
        elif k.startswith("decoder_transformer.transformer.layers."):
            r = k.replace("decoder_transformer.transformer.layers.", "decoder.decoder_transformer.layers.")
            if ".self_attn.in_proj.weight" in r:
                q, kk, vv = v.chunk(3, 0); b = r.replace("in_proj", "{}")
                perm = torch.cat([torch.arange(0, 64, 2), torch.arange(1, 64, 2)])     # interleaved rope pairs -> half-split pairs
                fix = lambda w: w.view(8, 64, -1)[:, perm, :].reshape(w.shape[0], -1)
                q, kk = (fix(q), fix(kk)) if permute else (q, kk)
                sd[b.format("q_proj")], sd[b.format("k_proj")], sd[b.format("v_proj")] = q, kk, vv; continue
            sd[(r.replace("self_attn.out_proj", "self_attn.o_proj").replace(".linear1.", ".mlp.fc1.").replace(".linear2.", ".mlp.fc2.")
                 .replace(".norm1.", ".input_layernorm.").replace(".norm2.", ".post_attention_layernorm.")
                 .replace("layer_scale_1.scale", "self_attn_layer_scale.scale").replace("layer_scale_2.scale", "mlp_layer_scale.scale"))] = v
        elif k == "quantizer.output_proj.weight": sd["latent_proj.weight"] = v
        elif k == "upsample.convtr.convtr.weight": sd["decoder.upsample.conv.weight"] = v
        elif k.startswith(("encoder.", "encoder_transformer.", "downsample.")): pass
        else: raise KeyError(k)
    m = PocketTTSAudioTokenizerModel(PocketTTSAudioTokenizerConfig(encoder_config=None, hidden_act=act)).eval()
    res = m.load_state_dict(sd, strict=False)
    print("mimi load: missing", [k for k in res.missing_keys if "rotary" not in k], "unexpected", res.unexpected_keys); return m

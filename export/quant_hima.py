import onnx, sys, os, time
from onnxruntime.quantization import matmul_nbits_quantizer as q
src, dst, bits, block = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
m = onnx.load(src)
z = q.MatMulNBitsQuantizer(m, algo_config=q.DefaultWeightOnlyQuantConfig(block_size=block, is_symmetric=True, bits=bits, accuracy_level=4))
z.process(); z.model.save_model_to_file(dst, use_external_data_format=False); print(dst, round(os.path.getsize(dst) / 1e6, 1), "MB")

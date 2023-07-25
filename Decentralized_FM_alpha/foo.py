import torch
import cupy

print("<== Load torch and cupy. ==>")
print("Torch Version: ", torch.__version__)
print("CuPy Version: ", cupy.__version__)
print('GPU available: ', torch.cuda.is_available())

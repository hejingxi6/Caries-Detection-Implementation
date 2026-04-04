import sys
import torch

print('===== Environment Check =====')
print(f'Python executable: {sys.executable}')
print(f'Python version: {sys.version}')
print('\n===== Torch Info =====')
print(f'Torch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
print('\n===== Done =====')

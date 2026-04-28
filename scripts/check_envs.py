"""Check correct installation of flash-attn"""

import torch


def check():
    print("--- Environment Report ---")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    try:
        import flash_attn
        print(f"Flash-Attn: {flash_attn.__version__}")
    except ImportError:
        print("Flash-Attn: Not installed or incompatible version.")
        

if __name__ == "__main__":
    check()
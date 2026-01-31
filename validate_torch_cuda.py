import sys

import torch


def main() -> int:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version (build): {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    print("Arch list:", torch.cuda.get_arch_list())

    if not torch.cuda.is_available():
        print("CUDA is not available. Check driver/CUDA install and that you installed a CUDA build of PyTorch.")
        return 1

    device_idx = 0
    device = torch.device(f"cuda:{device_idx}")
    name = torch.cuda.get_device_name(device_idx)
    cap = torch.cuda.get_device_capability(device_idx)
    print(f"Using device: {device}")
    print(f"Device name: {name}")
    print(f"Compute capability: {cap}")

    x = torch.randn(4096, 4096, device=device)
    y = torch.randn(4096, 4096, device=device)
    torch.cuda.synchronize()

    z = x @ y
    torch.cuda.synchronize()

    print(f"Output device: {z.device}")
    print(f"Output mean: {z.mean().item():.6f}")
    print("CUDA test completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

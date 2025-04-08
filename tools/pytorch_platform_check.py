import torch
import platform

def get_device():
    system = platform.system()
    
    if system == "Darwin":  # macOS
        # Check for M1/M2 chip Metal support
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device

if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")
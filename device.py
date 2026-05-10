"""device.py — Single source of truth for compute device across all scripts."""

def get_device():
    """Return (device, label) — prefers CUDA/ROCm, then DirectML, then CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            return "cuda", f"GPU — {name} ({vram:.1f} GB VRAM) [CUDA/ROCm]"
    except Exception:
        pass

    try:
        import torch_directml
        dml  = torch_directml.device()
        name = torch_directml.device_name(0)
        return dml, f"GPU — {name} [DirectML]"
    except (ImportError, Exception):
        pass

    return "cpu", "CPU (no GPU detected)"


DEVICE, DEVICE_LABEL = get_device()


def report():
    print(f"Compute device : {DEVICE_LABEL}")


if __name__ == "__main__":
    report()

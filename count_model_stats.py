import os
os.environ["XFORMERS_DISABLED"] = "1"

import torch
import inspect
from thop import profile, clever_format
from model import Model


def count_model_stats(model, input_size=(1, 3, 378, 378), device="cuda", verbose=True):
    """
    Count and print:
      - Total Params
      - Trainable Params
      - Frozen Params
      - Buffers
      - FLOPs
      - Parameter/Buffer/Total model size

    Args:
        model: PyTorch model
        input_size: input tensor size, e.g. (1, 3, 378, 378)
        device: "cuda" or "cpu"
        verbose: whether to print per-layer parameter info

    Returns:
        stats (dict): all computed statistics
    """
    model = model.to(device)
    model.eval()

    # =========================================================
    # 1) Count parameters
    # =========================================================
    total_params_count = 0
    trainable_params_count = 0
    frozen_params_count = 0

    total_param_size_bytes = 0
    total_buffer_count = 0
    total_buffer_size_bytes = 0

    if verbose:
        print(f"{'Layer Name':<60} | {'Count (M)':<12} | {'Status'}")
        print("-" * 100)

    for name, param in model.named_parameters():
        p_count = param.numel()
        p_size = p_count * param.element_size()

        total_params_count += p_count
        total_param_size_bytes += p_size

        if param.requires_grad:
            trainable_params_count += p_count
            status = "Trainable"
        else:
            frozen_params_count += p_count
            status = "Frozen"

        if verbose:
            print(f"{name:<60} | {p_count / 1e6:>10.4f} M | {status}")

    # =========================================================
    # 2) Count buffers separately
    # =========================================================
    for name, buf in model.named_buffers():
        b_count = buf.numel()
        b_size = b_count * buf.element_size()

        total_buffer_count += b_count
        total_buffer_size_bytes += b_size

        if verbose:
            print(f"{name:<60} | {b_count / 1e6:>10.4f} M | Buffer")

    # =========================================================
    # 3) Prepare dummy input(s)
    # =========================================================
    input1 = torch.randn(*input_size).to(device)
    input2 = torch.randn(*input_size).to(device)

    # =========================================================
    # 4) FLOPs with THOP
    #    Auto-handle 1-input or 2-input forward
    # =========================================================
    flops = None
    thop_params = None
    profile_mode = None

    try:
        # First try single-input profile
        flops, thop_params = profile(model, inputs=(input1,), verbose=False)
        profile_mode = "single-input"
    except Exception as e1:
        try:
            # If failed, try two-input profile
            flops, thop_params = profile(model, inputs=(input1, input2), verbose=False)
            profile_mode = "two-input"
        except Exception as e2:
            print("\n[Warning] THOP FLOPs profiling failed.")
            print("Single-input error:", e1)
            print("Two-input error   :", e2)
            flops, thop_params = None, None
            profile_mode = "failed"

    # =========================================================
    # 5) Format results
    # =========================================================
    total_params_m = total_params_count / 1e6
    trainable_params_m = trainable_params_count / 1e6
    frozen_params_m = frozen_params_count / 1e6
    buffer_count_m = total_buffer_count / 1e6

    param_size_mb = total_param_size_bytes / (1024 ** 2)
    buffer_size_mb = total_buffer_size_bytes / (1024 ** 2)
    total_model_size_mb = (total_param_size_bytes + total_buffer_size_bytes) / (1024 ** 2)

    if flops is not None and thop_params is not None:
        flops_fmt, thop_params_fmt = clever_format([flops, thop_params], "%.3f")
    else:
        flops_fmt, thop_params_fmt = "N/A", "N/A"

    # =========================================================
    # 6) Print summary
    # =========================================================
    print("\n" + "=" * 100)
    print("Model Statistics Summary")
    print("=" * 100)
    print(f"Profile Mode              : {profile_mode}")
    print(f"Total Parameters          : {total_params_m:.3f} M")
    print(f"Trainable Parameters      : {trainable_params_m:.3f} M")
    print(f"Frozen Parameters         : {frozen_params_m:.3f} M")
    print(f"Total Buffers             : {buffer_count_m:.3f} M")
    print(f"THOP Params (reference)   : {thop_params_fmt}")
    print(f"FLOPs                     : {flops_fmt}")
    print(f"Parameter Size            : {param_size_mb:.3f} MB")
    print(f"Buffer Size               : {buffer_size_mb:.3f} MB")
    print(f"Total Model Size          : {total_model_size_mb:.3f} MB")
    print("=" * 100)

    stats = {
        "profile_mode": profile_mode,
        "total_params": total_params_count,
        "trainable_params": trainable_params_count,
        "frozen_params": frozen_params_count,
        "buffer_count": total_buffer_count,
        "total_params_m": total_params_m,
        "trainable_params_m": trainable_params_m,
        "frozen_params_m": frozen_params_m,
        "buffer_count_m": buffer_count_m,
        "flops": flops,
        "thop_params": thop_params,
        "flops_fmt": flops_fmt,
        "thop_params_fmt": thop_params_fmt,
        "param_size_mb": param_size_mb,
        "buffer_size_mb": buffer_size_mb,
        "total_model_size_mb": total_model_size_mb,
    }

    return stats


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model(model_name='dinov2_vitb14')

    stats = count_model_stats(
        model=model,
        input_size=(1, 3, 378, 378),
        device=device,
        verbose=False
    )
import torch

def print_cuda_device_properties(device: int = 0):
    props = torch.cuda.get_device_properties(device)

    print("=" * 60)
    print(f"CUDA Device {device}")
    print("=" * 60)

    print(f"Name:                         {props.name}")
    print(f"UUID:                         {props.uuid}")
    print(f"Compute Capability:           {props.major}.{props.minor}")

    # Architecture info
    if hasattr(props, "gcnArchName"):
        print(f"GCN Arch Name:                {props.gcnArchName}")

    print(f"Integrated GPU:               {props.is_integrated}")
    print(f"Multi-GPU Board:              {props.is_multi_gpu_board}")

    print("-" * 60)

    # SM / execution model
    print(f"SM count:                     {props.multi_processor_count}")
    print(f"Max threads / SM:             {props.max_threads_per_multi_processor}")
    print(f"Warp size:                    {props.warp_size}")

    print("-" * 60)

    # Memory hierarchy
    print(f"Total global memory:          {props.total_memory / 1024**3:.2f} GB")
    print(f"L2 cache size:                {props.L2_cache_size / 1024:.0f} KB")

    print(f"Shared memory / SM:           {props.shared_memory_per_multiprocessor / 1024:.0f} KB")
    print(f"Shared memory / block:        {props.shared_memory_per_block / 1024:.0f} KB")
    print(f"Shared memory / block(optin): {props.shared_memory_per_block_optin / 1024:.0f} KB")

    print("-" * 60)

    # Registers
    print(f"Registers / SM:               {props.regs_per_multiprocessor}")

    print("=" * 60)

print_cuda_device_properties()

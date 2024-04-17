import torch

# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available")
    print("Number of GPUs:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print("GPU", i, ":", torch.cuda.get_device_name(i))
        print(
            "Total memory for GPU",
            i,
            ":",
            torch.cuda.get_device_properties(i).total_memory / (1024**3),
            "GB",
        )
        print("Compute capability for GPU", i, ":", torch.cuda.get_device_capability(i))
else:
    print("GPU is not available")

#!/usr/bin/env python3
import sys
import os
import json
from pathlib import Path

import numpy as np
from gguf import GGUFReader, ReaderTensor


def sanitize_name(name: str) -> str:
    """
    Tensor name to file-safe name.
    You could also replace only spaces, but this way you're safe.
    """
    return name.replace("/", "_").replace(" ", "_")


def numerical_sort_key(tensor_name: str) -> list:
    """
    Generate a key for numerical sorting of tensor names.
    Splits the name into parts and converts numeric parts to integers.
    """
    import re
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', tensor_name)]


def print_test_tensor(tensor: ReaderTensor) -> None:
    """
    Print information and first 10 bytes about of a tensor.
    """
    print(
        f"Test tensor: {tensor.name}, dtype: {tensor.tensor_type.name}")
    print("First 10 bytes:")
    bytes_data = tensor.data.flatten()
    for byte in bytes_data[:10]:
        print(f"0x{byte:02x}", end=" ")
    print("")


def main(gguf_path: Path, weights_dir: Path, index_path: Path) -> None:
    weights_dir.mkdir(parents=True, exist_ok=True)

    reader = GGUFReader(str(gguf_path))

    index = {
        "source": str(gguf_path),
        "tensors": []
    }

    # Sort tensors numerically by their names
    sorted_tensors = sorted(reader.tensors, key=lambda t: numerical_sort_key(t.name))

    for tensor in sorted_tensors:
        name = tensor.name
        data = tensor.data  # np.ndarray
        dtype = tensor.tensor_type.name

        flat = data.flatten()

        filename = sanitize_name(name) + f"_{dtype}.bin"
        out_path = weights_dir / filename

        flat.tofile(out_path)

        # add to the JSON index
        index["tensors"].append({
            "name": name,
            "dtype": tensor.tensor_type.name,
            "shape": list(data.shape),
            "filename": filename
        })

    # write the index
    print(f"Tensors saved in {weights_dir}")
    with index_path.open("w") as f:
        json.dump(index, f, indent=2)
    print(f"Index written to: {index_path}")

    test_tensor = sorted_tensors[0]
    print_test_tensor(test_tensor)


if __name__ == "__main__":
    base_dir = Path("granite-4.0-350m-BF16")
    gguf_path = base_dir / "model.gguf"
    weights_dir = base_dir / "weights"
    index_path = base_dir / "weights_index.json"

    main(gguf_path, weights_dir, index_path)

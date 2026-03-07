import numpy as np
from pathlib import Path

DATA_FOLDER = '/scratch/Projects/CFP-01/CFP01-CF-060/kieron/data/fineweb2_SEA_100M_sentences_parity-aware'
DTYPE = np.uint32

def main():
    # Get all .npy files in the folder
    data_folder_path = Path(DATA_FOLDER)
    data_files = sorted(data_folder_path.glob('*.npy'))

    if not data_files:
        print(f"No .npy files found in {DATA_FOLDER}")
        return

    total_tokens = 0
    print(f"Calculating bytes for {len(data_files)} files...")

    for i, file_path in enumerate(data_files):
        if not file_path.exists():
            print(f"[Missing] {file_path}")
            continue

        try:
            file_size_bytes = file_path.stat().st_size
            itemsize = np.dtype(DTYPE).itemsize

            if file_size_bytes % itemsize != 0:
                print(f"[Warning] File size not aligned for {file_path}")

            current_count = file_size_bytes // itemsize
            total_tokens += current_count

            if i % 1000 == 0:
                print(f"Processed {i+1}/{len(data_files)} | Current Total: {total_tokens}")

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    print(f"\nFinal Total Bytes: {total_tokens}")
    print(f"Average bytes per file: {total_tokens / len(data_files):.2f}")


if __name__ == "__main__":
    main()

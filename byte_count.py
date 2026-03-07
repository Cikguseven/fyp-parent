from typing import List
from pathlib import Path
import concurrent.futures
import os
# Removed tokenizer imports

def read_lines(fp):
    lines: List[str] = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def count_utf8_bytes(lines):
    """Counts the total UTF-8 bytes in the list of strings."""
    if not lines:
        return 0
    # Encode each line to utf-8 and sum the lengths
    return sum(len(line.encode("utf-8")) for line in lines)


def process_file(path_str):
    # Removed model_path argument
    p = Path(path_str)
    lines = read_lines(p)
    # Using byte count instead of token count
    c = count_utf8_bytes(lines)
    return str(p), c


def main(
    folder,
):
    root = Path(folder)
    files = sorted(list(root.glob("*.txt")))

    total = 0
    per_file = []

    max_workers = min(len(files) or 1, os.cpu_count() or 1)
    print(f"Processing {len(files)} files with {max_workers} processes...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Removed model_path from arguments
        futures = [executor.submit(process_file, str(p)) for p in files]

        for future in concurrent.futures.as_completed(futures):
            p, c = future.result()
            print(f"Processed {p}: {c}")
            per_file.append((p, c))
            total += c

    per_file.sort()
    for p, c in per_file:
        print(f"{c}\t{p}")
    print(f"\nTOTAL\t{total}")


if __name__ == "__main__":
    main(
        # Removed model_path
        folder="/scratch/Projects/CFP-01/CFP01-CF-060/kieron/data/mc4_SEA_1M_sentences"
    )

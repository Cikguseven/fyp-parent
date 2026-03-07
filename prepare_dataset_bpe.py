"""
Use this to prepare a numpy memory-mapped language modeling dataset from raw *.txt
dataset files. Each file is expected to be a plain text file representing a
single document (or a chunk of text) from the dataset.

This script supports custom BPE tokenizers defined by a tokenizer.json file.

Usage:
    python prepare_memmap_txt.py \
        /path/to/txt/files \
        -o /path/to/output_dir \
        --tokenizer-file /path/to/tokenizer.json \
        --eos-token-id 0 \
        --workers 8
"""

import concurrent.futures
import functools
import logging
import multiprocessing as mp
import os
import random
import json
import glob
from concurrent.futures import Future
from contextlib import ExitStack
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Generator, List, Optional, Sequence, Tuple, TypeVar, Union, Set

import click
import numpy as np
from tokenizers import Tokenizer
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TimeElapsedColumn,
    SpinnerColumn,
    TextColumn
)
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

from smashed.utils.io_utils import (
    MultiPath,
    open_file_for_write,
    recursively_list_files,
    stream_file_for_read,
)

# Standard logging setup
console = Console(theme=Theme({"logging.level.info": "cyan"}))

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)]
)
log = logging.getLogger("prepare_dataset")

T = TypeVar("T", bound=Sequence)

def get_progress() -> Progress:
    """Returns a Rich Progress bar configured to work with the shared console."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,  # IMPORTANT: Share the console instance
        transient=False   # Keep the bar visible after completion? (False = yes)
    )

def tokenize_file(
    tokenizer: Tokenizer,
    path: str,
    eos_token_id: Optional[int] = 90369,
    batch_size: int = 10000
) -> Generator[List[int], None, None]:
    """
    Tokenize a plain text file using streaming and batching.
    """
    batch = []

    try:
        # Open in text mode 'rt' to read line by line
        with stream_file_for_read(path, mode="rt") as f:
            for line in f:
                text = line.strip()
                if text:
                    batch.append(text)

                    # Process when batch is full
                    if len(batch) >= batch_size:
                        # encode_batch is faster (uses Rust parallelism)
                        encodings = tokenizer.encode_batch(batch, add_special_tokens=True)
                        for enc in encodings:
                            ids = enc.ids
                            if eos_token_id is not None:
                                ids.append(eos_token_id)
                            yield ids

                        batch = []

            # Process remaining items in the last batch
            if batch:
                encodings = tokenizer.encode_batch(batch, add_special_tokens=True)
                for enc in encodings:
                    ids = enc.ids
                    if eos_token_id is not None:
                        ids.append(eos_token_id)
                    yield ids

    except Exception as e:
        log.error(f"Error processing {path} -> {e}")
        pass


class MemmapFile:
    """Context manager responsible for writing, resizing, and closing / uploading a memmap file."""

    DEFAULT_MAX_TOKENS = 262144

    def __init__(
        self,
        path: str,
        dtype: np.dtype,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.path = MultiPath.parse(path)
        self.dtype = dtype
        self.max_tokens = max_tokens

        self._local_path: Optional[Path] = None
        self._written_tokens = 0
        self._memmap: Optional[np.memmap] = None

    def __len__(self) -> int:
        return self._written_tokens

    def write(self, values: List[int], flush: bool = False) -> Optional[List[int]]:
        if self._memmap is None:
            raise RuntimeError("MemmapFile is not open")

        if (len(values) + self._written_tokens) >= self.max_tokens:
            values_to_write = values[: self.max_tokens - self._written_tokens]
            rest = values[self.max_tokens - self._written_tokens :]
        else:
            values_to_write = values
            rest = None

        self._memmap[self._written_tokens : self._written_tokens + len(values_to_write)] = values_to_write
        self._written_tokens += len(values_to_write)

        if flush:
            self._memmap.flush()

        return rest

    def __enter__(self) -> "MemmapFile":
        assert self._memmap is None, "MemmapFile is already open"

        if self.path.is_local:
            self._local_path = self.path.as_path
            self._local_path.parent.mkdir(parents=True, exist_ok=True)  # type: ignore
        else:
            with NamedTemporaryFile(delete=False, prefix="olmo_memmap") as f:
                self._local_path = Path(f.name)

        self._memmap = np.memmap(mode="w+", filename=self._local_path, dtype=self.dtype, shape=(self.max_tokens,))
        return self

    def __exit__(self, *_):
        return self.close()

    def close(self):
        assert self._local_path is not None, "MemmapFile is not open"
        assert self._memmap is not None, "MemmapFile is not open"

        try:
            self._memmap.flush()

            # Resize to actual content
            if self._written_tokens < self.max_tokens:
                del self._memmap
                temp_path = self._local_path.with_suffix(".tmp")
                os.rename(self._local_path, temp_path)

                new_memmap = np.memmap(
                    mode="w+", filename=self._local_path, dtype=self.dtype, shape=(self._written_tokens,)
                )
                old_memmap = np.memmap(mode="r", filename=temp_path, dtype=self.dtype, shape=(self.max_tokens,))
                new_memmap[:] = old_memmap[: self._written_tokens]
                new_memmap.flush()
                os.remove(temp_path)

            if not self.path.is_local:
                with ExitStack() as stack:
                    f = stack.enter_context(stream_file_for_read(self._local_path, "rb"))
                    g = stack.enter_context(open_file_for_write(self.path, mode="wb"))
                    g.write(f.read())
                log.info(f"Written memmap file to {self.path.as_str}")
        finally:
            if not self.path.is_local:
                os.remove(self._local_path)

        self._local_path = self._memmap = None

def fill_memmap(
    tokenizer_file: str,
    path_or_paths: Union[str, List[str]],
    memmap_path: str,
    dtype: np.dtype,
    eos_token_id: Optional[int] = 90369,
    max_tokens: int = 262144,
    random_seed: int = 3920,
) -> Tuple[int, List[str]]:
    """
    Write a memmap file from a file of documents.
    Returns: (total_tokens_written, list_of_processed_files)
    """

    np.random.seed(random_seed)

    tokenizer = Tokenizer.from_file(tokenizer_file)

    memmap: Optional[MemmapFile] = None
    file_index = 0
    total_tokens = 0

    path_or_paths = [path_or_paths] if isinstance(path_or_paths, str) else path_or_paths
    processed_files = []

    with ExitStack() as stack:
        for single_path in path_or_paths:
            try:
                it = tokenize_file(tokenizer=tokenizer, path=single_path, eos_token_id=eos_token_id)

                for line_no, token_ids in enumerate(it, start=1):
                    flush = line_no % 10000 == 0
                    total_tokens += len(token_ids)

                    leftovers_to_write = memmap.write(token_ids, flush=flush) if memmap is not None else token_ids

                    if leftovers_to_write is not None:
                        if memmap is not None:
                            stack.pop_all().close()

                        # Note: memmap_path already has the base index (e.g. .../00001)
                        # We append specific parts to it like .../00001_00000.npy if multiple parts needed
                        curr_memmap_path = f"{memmap_path}_{file_index:05d}.npy"
                        memmap = stack.enter_context(MemmapFile(path=curr_memmap_path, dtype=dtype, max_tokens=max_tokens))
                        file_index += 1
                        memmap.write(leftovers_to_write)

                # If we finished the loop for this file without exception, mark it processed
                processed_files.append(single_path)

            except Exception as e:
                log.error(f"Failed to process {single_path}: {e}")
                # We do NOT add it to processed_files, so it will be retried next time.

        if memmap is not None:
            stack.pop_all().close()

    return total_tokens, processed_files


def get_next_output_index(output_dir: str) -> int:
    """Scans output directory for existing .npy files and returns the next safe index."""
    if not os.path.exists(output_dir):
        return 0

    # Look for files matching format *_xxxxx.npy
    files = glob.glob(os.path.join(output_dir, "*.npy"))
    if not files:
        return 0

    max_idx = 0
    # Try to guess structure. Usually /path/to/00001_00000.npy
    for f in files:
        try:
            basename = os.path.basename(f)
            parts = basename.split('_')
            if len(parts) >= 2:
                idx_str = parts[0]
                idx = int(idx_str)
                if idx > max_idx:
                    max_idx = idx
        except ValueError:
            continue

    return max_idx + 1


def make_source_and_target(
    src: Tuple[str, ...],
    output: str,
    processed_files: Set[str],
    start_index: int = 0,
    random_seed: int = 3920,
    paths_per_worker: int = 1,
) -> Tuple[Tuple[Union[str, List[str]], ...], Tuple[str, ...]]:

    np.random.seed(random_seed)
    random.seed(random_seed)

    # Filter strictly for .txt files
    all_files = []
    for prefix in src:
        found = recursively_list_files(prefix)
        # Filter out already processed files
        for p in found:
            p_str = str(p)
            if p_str.endswith('.txt') and p_str not in processed_files:
                all_files.append(p_str)

    exploded_src = sorted(list(set(all_files)))

    if not exploded_src:
        return (), ()

    output_digits = 5
    random.shuffle(exploded_src)

    if paths_per_worker > 1:
        exploded_src = [
            sorted(exploded_src[i : i + paths_per_worker]) for i in range(0, len(exploded_src), paths_per_worker)
        ]

    # Generate destinations starting from the existing offset
    exploded_dst = [f'{output.rstrip("/")}/{i + start_index:0{output_digits}d}' for i in range(len(exploded_src))]

    return tuple(exploded_src), tuple(exploded_dst)


@click.command()
@click.argument(
    "src",
    nargs=-1,
    type=str,
    required=True,
)
@click.option(
    "-o",
    "--output",
    type=str,
    help="Specify the output path.",
    required=True,
)
@click.option(
    "--tokenizer-file",
    type=str,
    help="Path to tokenizer.json file",
    required=True,
)
@click.option(
    "--eos-token-id",
    type=int,
    default=90369,
    help="Explicitly specify EOS token ID to append to every document.",
)
@click.option("--dtype", "dtype_str", default="uint32")
@click.option("--validate/--no-validate", default=False)
@click.option("--random-seed", type=int, default=3920)
@click.option("--paths-per-worker", type=click.IntRange(min=1), default=1)
@click.option(
    "--max-tokens",
    default=262144,
    type=int,
    help="Maximum number of tokens per memmap file (default: 262144 tokens)",
)
@click.option("--debug/--no-debug", default=False, help="Enable debug (single process mode)")
@click.option("-j", "--workers", "max_workers", type=int, default=1, help="Number of worker processes")
def main(
    src: Tuple[str, ...],
    output: str,
    tokenizer_file: str,
    eos_token_id: Optional[int] = 90369,
    dtype_str: str = "uint32",
    validate: bool = False,
    max_tokens: int = 262144,
    debug: bool = False,
    random_seed: int = 3920,
    paths_per_worker: int = 1,
    max_workers: int = 1,
):
    print("=== CONFIGURATION ===")
    print(f"src:              {src}")
    print(f"output:           {output}")
    print(f"tokenizer_file:   {tokenizer_file}")
    print(f"eos_token_id:     {eos_token_id}")
    print(f"dtype_str:        {dtype_str}")
    print(f"max_tokens:       {max_tokens}")
    print(f"max_workers:      {max_workers}")
    print("=====================")

    dtype = np.dtype(dtype_str)

    # 0. Setup Manifest
    os.makedirs(output, exist_ok=True)
    manifest_path = os.path.join(output, "manifest.json")
    processed_files: Set[str] = set()

    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r') as f:
                data = json.load(f)
                processed_files = set(data.get("files", []))
            log.info(f"Loaded manifest. Found {len(processed_files)} previously processed files.")
        except Exception as e:
            log.warning(f"Could not load manifest: {e}. Starting fresh.")

    # 1. Determine start index
    start_index = get_next_output_index(output)
    log.info(f"Starting output file indexing at: {start_index}")

    # 2. Gather files
    exploded_src, exploded_dst = make_source_and_target(
        src=src,
        output=output,
        processed_files=processed_files,
        start_index=start_index,
        random_seed=random_seed,
        paths_per_worker=paths_per_worker
    )

    if not exploded_src:
        log.info("No new .txt files found to process! (Check manifest if you expected files)")
        return

    # Count real files
    total_files = sum(len(x) if isinstance(x, list) else 1 for x in exploded_src)
    log.info(f"Found {total_files} new source files.")

    # 3. Prepare worker
    fill_memmap_fn = functools.partial(
        fill_memmap,
        tokenizer_file=tokenizer_file,
        dtype=dtype,
        eos_token_id=eos_token_id,
        max_tokens=max_tokens,
        random_seed=random_seed,
    )

    total_tokens_written = 0

    # 4. Execution
    if debug:
        log.info("Running in debug mode. Only one process will be used.")
        for src_path, dst_path in zip(exploded_src, exploded_dst):
            tokens, p_files = fill_memmap_fn(path_or_paths=src_path, memmap_path=dst_path)
            total_tokens_written += tokens

            processed_files.update(p_files)
            with open(manifest_path, 'w') as f:
                json.dump({"files": list(processed_files)}, f)

    else:
        workers_cnt = min(max_workers or os.cpu_count() or 1, len(exploded_src))

        # NOTE: Using a shared console in the context manager
        with get_progress() as progress:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers_cnt) as executor:
                futures: List[Future[Tuple[int, List[str]]]] = []

                # Submit all jobs
                for src_path, dst_path in zip(exploded_src, exploded_dst):
                    future = executor.submit(fill_memmap_fn, path_or_paths=src_path, memmap_path=dst_path)
                    futures.append(future)

                # Track completion
                task_id = progress.add_task("Processing batches...", total=len(futures))

                for future in concurrent.futures.as_completed(futures):
                    try:
                        tokens, p_files = future.result()
                        total_tokens_written += tokens

                        # Update progress bar
                        progress.advance(task_id)

                        # Update Manifest
                        if p_files:
                            processed_files.update(p_files)
                            # Simple atomic-ish dump
                            with open(manifest_path, 'w') as f:
                                json.dump({"files": list(processed_files)}, f)

                    except Exception as e:
                        log.error(f"Worker failed: {e}")

    log.info(f"Done! File(s) written to {output}")
    log.info(f"Total tokens written: {total_tokens_written:,}")
    log.info(f"Manifest updated at {manifest_path}")

    # 5. Validation
    if validate:
        log.info("Validating...")
        # Re-initialize tokenizer for validation
        tokenizer = Tokenizer.from_file(tokenizer_file)

        # Simplified validation that checks counts without keeping everything in RAM
        total_tokens_check = 0
        all_src_flat = []
        for item in exploded_src:
            if isinstance(item, list):
                all_src_flat.extend(item)
            else:
                all_src_flat.append(item)

        for input_path in all_src_flat:
             with stream_file_for_read(input_path, mode="rt") as f:
                for line in f:
                    text = line.strip()
                    if text:
                        ids = tokenizer.encode(text, add_special_tokens=True).ids
                        if eos_token_id is not None:
                            ids.append(eos_token_id)
                        total_tokens_check += len(ids)

        total_tokens_on_disk = 0
        output_files = sorted([str(p) for p in recursively_list_files(output) if str(p).endswith('.npy')])

        for output_path in output_files:
            memmap = np.memmap(output_path, mode="r", dtype=dtype)
            total_tokens_on_disk += len(memmap)

        log.info(f"Source tokens (Current Batch): {total_tokens_check:,}")
        log.info(f"Disk tokens (Total Output):    {total_tokens_on_disk:,}")

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
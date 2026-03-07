import os
import glob
import time
import random
import datasets
import huggingface_hub
from multiprocessing import Pool
from tqdm.auto import tqdm
import multiprocessing

# ---------------- CONFIGURATION ----------------
huggingface_hub.login(token=os.getenv("HF_TOKEN"))

N_TOTAL = 100_000_000
MAX_BYTES_PER_FILE = 256 * 1024 * 1024
SEED = 42  # Global random seed for reproducibility

OUT_DIR = "/scratch/Projects/CFP-01/CFP01-CF-060/kieron/data/fineweb2_SEA_100M_sentences"
os.makedirs(OUT_DIR, exist_ok=True)

# --- WORKER CONFIGURATION ---
MAX_GLOBAL_WORKERS = 10
EN_SHARDS = 9       # 9 workers dedicated to English

SEA_LANGS = ["en", "zh", "id", "vi", "th", "ms", "ta", "fil", "my", "km", "lo"]

SEA_langs_prop = {
    "en": 89.689566, "zh": 7.199747, "id": 1.169277, "vi": 1.016743,
    "th": 0.558780, "ms": 0.165279, "ta": 0.068252, "fil": 0.059364,
    "my": 0.034699, "km": 0.028295, "lo": 0.009998,
}

# NOTE: To make this TRULY deterministic forever, replace "main" with the specific
# commit hash (SHA) from the dataset's Hugging Face 'History' tab.
# Example: "revision": "3e201..."
LANG_CONFIG_MAP = {
    "en":  {"dataset": "HuggingFaceFW/fineweb",   "config": "CC-MAIN-2025-26", "revision": "9bb295ddab0e05d785b879661af7260fed5140fc"},
    "zh":  {"dataset": "HuggingFaceFW/fineweb-2", "config": "cmn_Hani",        "revision": "af9c13333eb981300149d5ca60a8e9d659b276b9"},
    "id":  {"dataset": "HuggingFaceFW/fineweb-2", "config": "ind_Latn",        "revision": "af9c13333eb981300149d5ca60a8e9d659b276b9"},
    "vi":  {"dataset": "HuggingFaceFW/fineweb-2", "config": "vie_Latn",        "revision": "af9c13333eb981300149d5ca60a8e9d659b276b9"},
    "th":  {"dataset": "HuggingFaceFW/fineweb-2", "config": "tha_Thai",        "revision": "af9c13333eb981300149d5ca60a8e9d659b276b9"},
    "ms":  {"dataset": "HuggingFaceFW/fineweb-2", "config": "zsm_Latn",        "revision": "af9c13333eb981300149d5ca60a8e9d659b276b9"},
    "ta":  {"dataset": "HuggingFaceFW/fineweb-2", "config": "tam_Taml",        "revision": "af9c13333eb981300149d5ca60a8e9d659b276b9"},
    "fil": {"dataset": "HuggingFaceFW/fineweb-2", "config": "fil_Latn",        "revision": "af9c13333eb981300149d5ca60a8e9d659b276b9"},
    "my":  {"dataset": "HuggingFaceFW/fineweb-2", "config": "mya_Mymr",        "revision": "af9c13333eb981300149d5ca60a8e9d659b276b9"},
    "km":  {"dataset": "HuggingFaceFW/fineweb-2", "config": "khm_Khmr",        "revision": "af9c13333eb981300149d5ca60a8e9d659b276b9"},
    "lo":  {"dataset": "HuggingFaceFW/fineweb-2", "config": "lao_Laoo",        "revision": "af9c13333eb981300149d5ca60a8e9d659b276b9"},
}

# ---------------- CALCULATION ----------------
targets = {}
total_weight = sum(SEA_langs_prop.get(lang, 0) for lang in SEA_LANGS)
remaining_budget = N_TOTAL

# Sort to ensure target calculation is identical every time
sorted_langs = sorted(SEA_LANGS, key=lambda l: SEA_langs_prop.get(l, 0), reverse=True)

for i, lang in enumerate(sorted_langs):
    weight = SEA_langs_prop.get(lang, 0)
    count = int((weight / total_weight) * N_TOTAL) if total_weight > 0 else 0
    if i == len(sorted_langs) - 1:
        count = remaining_budget
    targets[lang] = count
    remaining_budget -= count

print("Target document counts per language:")
for lang, count in targets.items():
    print(f"  {lang}: {count}")

# ---------------- CORE LOGIC ----------------
def get_shard_filename(lang_code, worker_id, shard_idx):
    return os.path.join(OUT_DIR, f"{lang_code}_w{worker_id}_{shard_idx}.txt")


def _download_lang(lang_code, target_count, worker_id, num_shards):
    """
    Internal function to download a single language/shard.
    """
    # Random sleep only affects timing, not data order.
    # Using local random instance to avoid affecting global state if it mattered.
    local_rng = random.Random(worker_id)
    time.sleep(local_rng.uniform(5.0, 15.0))

    if target_count <= 0:
        return lang_code, worker_id, 0, "SKIPPED"

    ds_info = LANG_CONFIG_MAP.get(lang_code)

    # --- Check Existing Progress ---
    existing_pattern = os.path.join(OUT_DIR, f"{lang_code}_w{worker_id}_*.txt")
    files = sorted(glob.glob(existing_pattern)) # Sorted ensures strict order checking

    written_so_far = 0
    current_shard_idx = 0
    current_shard_bytes = 0

    for fpath in files:
        with open(fpath, "rb") as f:
            while chunk := f.read(1 << 20):
                written_so_far += chunk.count(b"\n")

        # Only check size of the last file to see if we continue writing to it
        if fpath == files[-1]:
            f_size = os.path.getsize(fpath)
            try:
                base = os.path.splitext(os.path.basename(fpath))[0]
                current_shard_idx = int(base.split("_")[-1])
            except:
                current_shard_idx = 0

            if f_size < MAX_BYTES_PER_FILE:
                current_shard_bytes = f_size
            else:
                current_shard_idx += 1
                current_shard_bytes = 0

    remaining_target = target_count - written_so_far
    if remaining_target <= 0:
        return lang_code, worker_id, written_so_far, "DONE_ALREADY"

    # --- Load Dataset ---
    try:
        # Load dataset with pinned revision for determinism
        ds = datasets.load_dataset(
            ds_info["dataset"],
            name=ds_info["config"],
            split="train",
            streaming=True,
            revision=ds_info.get("revision", "main") # CRITICAL FOR DETERMINISM
        )

        # Deterministic Sharding
        # For streaming datasets, this typically splits the underlying file list.
        # As long as num_shards and worker_id are constant, this is deterministic.
        if num_shards > 1:
            ds = ds.shard(num_shards=num_shards, index=worker_id)

        # Resume from exact position
        if written_so_far > 0:
            ds = ds.skip(written_so_far)

    except Exception as e:
        return lang_code, worker_id, written_so_far, f"LOAD_ERROR: {e}"

    # --- Write Loop ---
    out_path = get_shard_filename(lang_code, worker_id, current_shard_idx)
    mode = "ab" if current_shard_bytes > 0 else "wb"
    f = open(out_path, mode)

    local_written = 0

    print(f"[{lang_code}-w{worker_id}] Fast-forwarding {written_so_far:,} records. This may take a while...")

    try:
        pbar = tqdm(
            total=remaining_target,
            desc=f"en-w{worker_id}",
            miniters=50_000,           # only update every 50k docs
            maxinterval=float("inf"),  # disable time-based forced updates
            position=worker_id
        )

        for ex in ds:
            pbar.update(1)
            text = ex.get("text")
            if not isinstance(text, str):
                continue

            clean_text = " ".join(text.splitlines()).strip()
            if not clean_text:
                continue

            line_bytes = (clean_text + "\n").encode("utf-8")
            len_bytes = len(line_bytes)

            if current_shard_bytes + len_bytes > MAX_BYTES_PER_FILE:
                f.close()
                current_shard_idx += 1
                out_path = get_shard_filename(lang_code, worker_id, current_shard_idx)
                f = open(out_path, "wb")
                current_shard_bytes = 0

            f.write(line_bytes)
            current_shard_bytes += len_bytes

            local_written += 1
            written_so_far += 1

            if local_written >= remaining_target:
                break

    except Exception as e:
        print(f"[{lang_code}-w{worker_id}] Error: {e}")
    finally:
        f.close()
        pbar.close()

    return lang_code, worker_id, written_so_far, "FINISHED"


# --- TASK WRAPPERS ---
def task_english(args):
    """Wrapper for English workers"""
    lang, target, worker_id, shards = args
    return _download_lang(lang, target, worker_id, shards)


def task_others(list_of_langs_args):
    """Wrapper for single 'Other Langs' worker. Processes sequentially."""
    results = []
    print(f"[Other-Worker] Starting sequential processing of {len(list_of_langs_args)} languages...")

    # Order of list_of_langs_args is fixed in main(), ensuring execution order
    for args in list_of_langs_args:
        lang, target, worker_id, shards = args
        print(f"[Other-Worker] Starting {lang} (Target: {target:,})...")
        res = _download_lang(lang, target, worker_id, shards)
        print(f"[Other-Worker] Finished {lang}: {res[3]}")
        results.append(res)

    total_docs = sum(r[2] for r in results)
    return "OTHERS", 0, total_docs, "ALL_FINISHED"


# ---------------- MAIN ----------------
def main():
    # Set seeds for reproducibility (though mostly logic-driven here)
    random.seed(SEED)

    print(f"Writing to: {OUT_DIR}")
    print(f"Configuration: {EN_SHARDS} English Workers | 1 General Worker")

    # 1. Prepare English Tasks
    en_tasks = []
    en_total = targets["en"]
    chunk_size = en_total // EN_SHARDS

    # w_id sequence is deterministic 0..8
    for w_id in range(EN_SHARDS):
        my_target = chunk_size + (1 if w_id < (en_total % EN_SHARDS) else 0)
        en_tasks.append(("en", my_target, w_id, EN_SHARDS))

    # 2. Prepare "Other" Tasks (grouped for 1 worker)
    # SEA_LANGS list is fixed order
    other_langs_args = []
    for lang in SEA_LANGS:
        if lang == "en":
            continue
        target = targets.get(lang, 0)
        other_langs_args.append((lang, target, 0, 1))

    # 3. Launch with Pool
    print(f"Launching {EN_SHARDS} English processes + 1 Other process...")
    print(f"Max concurrent workers: {MAX_GLOBAL_WORKERS}")

    with Pool(processes=MAX_GLOBAL_WORKERS, maxtasksperchild=1) as pool:
        results = []

        # Submit English tasks (Order of submission doesn't affect data content)
        for args in en_tasks:
            result = pool.apply_async(task_english, (args,))
            results.append(result)

        # Submit all other langs as ONE task
        result = pool.apply_async(task_others, (other_langs_args,))
        results.append(result)

        # Monitor with progress bar
        for r in tqdm(results, desc="Workers", total=len(results)):
            try:
                lang, worker_id, written, status = r.get(timeout=7200)  # 2 hour timeout
                if lang == "OTHERS":
                    tqdm.write(f"[Other Languages] Completed: {written:,} total docs")
                else:
                    tqdm.write(f"[{lang}-w{worker_id}] {status} (Docs: {written:,})")
            except Exception as e:
                tqdm.write(f"Worker failed: {e}")

    print("\nAll workers completed!")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
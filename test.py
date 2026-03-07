from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/scratch/Projects/CFP-01/CFP01-CF-060/kieron/8192_myte_SEA_1m", trust_remote_code=True)
print(tok.eos_token_id)  # should match TOKENIZER_DEFAULTS
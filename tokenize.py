import os

# --- INPUTS ---
VOCAB_FILE = "granite-4.0-350m-BF16/vocab.txt"
SYS_PROMPT = "You are a helpful assistant. Please ensure responses are professional, accurate, and safe."
USER_PROMPT = "Hello"

def load_vocab(filepath):
    vocab = {}
    specials = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                # Remove line break
                token = line.rstrip('\n')

                # Map token string to ID
                if token not in vocab:
                    vocab[token] = idx

                # Identify special tokens
                if token in ["<|start_of_role|>", "<|end_of_role|>", "<|end_of_text|>"]:
                    specials[token] = idx
                if token == 'Ċ': # The special character for Newline in BPE
                    specials['newline'] = idx
    except FileNotFoundError:
        print(f"Error: '{filepath}' not found.")
        exit(1)
    return vocab, specials

def greedy_encode(text, vocab):
    """Simulates BPE tokenization using greedy matching."""
    # Replace space with G token (U+0120)
    # Replace newline with C token (U+010A)
    text = text.replace(" ", "Ġ").replace("\n", "Ċ")

    tokens = []
    i = 0
    n = len(text)

    while i < n:
        match_id = None
        match_len = 0
        match_str = ""

        # Look for the longest possible token starting at i
        for j in range(n, i, -1):
            sub = text[i:j]
            if sub in vocab:
                match_id = vocab[sub]
                match_len = j - i
                match_str = sub
                break

        if match_id is not None:
            tokens.append((match_id, match_str))
            i += match_len
        else:
            # Fallback for unknown chars (byte fallback is complex, skipping for simplicity)
            i += 1

    return tokens

def print_c_block(role_name, content, vocab, specials, is_generation_trigger=False):

    # 1. Start Role
    sid = specials.get("<|start_of_role|>", 0)
    print(f"    tokens[length++] = {sid};".ljust(30) + " // <|start_of_role|>")

    # 2. Role Name
    role_tokens = greedy_encode(role_name, vocab)
    for tid, tstr in role_tokens:
        print(f"    tokens[length++] = {tid};".ljust(30) + f" // \"{tstr}\"")

    # 3. End Role
    eid = specials.get("<|end_of_role|>", 0)
    print(f"    tokens[length++] = {eid};".ljust(30) + " // <|end_of_role|>")

    # 4. Message Content
    msg_tokens = greedy_encode(content, vocab)
    for tid, tstr in msg_tokens:
        # Clean string for display (replace G with space visual)
        clean_str = tstr.replace('Ċ', '\\n')
        print(f"    tokens[length++] = {tid};".ljust(30) + f" // \"{clean_str}\"")

    # 5. End of Text + Newline (Template Requirement)
    eot = specials.get("<|end_of_text|>", 0)
    nl = specials.get("newline", 198)
    print(f"    tokens[length++] = {eot};".ljust(30) + " // <|end_of_text|>")
    print(f"    tokens[length++] = {nl};".ljust(30) + " // \\n")

def main():
    if not os.path.exists(VOCAB_FILE):
        print(f"Please ensure {VOCAB_FILE} is in this folder.")
        return

    vocab, specials = load_vocab(VOCAB_FILE)

    print("    // Copy this into your main.c")
    print("")
    print(f"    // --- Standard chat sysprompt ---")

    # 1. System
    print_c_block("system", SYS_PROMPT, vocab, specials)

    # 2. User
    print_c_block("user", USER_PROMPT, vocab, specials)

    # 3. Assistant Trigger
    print_c_block("assistant", "", vocab, specials, is_generation_trigger=True)

if __name__ == "__main__":
    main()
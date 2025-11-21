from pathlib import Path
from transformers import AutoTokenizer


def main(model_path: str, out_path: str) -> None:
    tok = AutoTokenizer.from_pretrained(model_path)
    vocab = tok.get_vocab()  # dict: token -> id

    # invertiamo: id -> token
    id_to_token = {idx: tok for tok, idx in vocab.items()}
    max_id = max(id_to_token.keys())

    out_path = Path(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(max_id + 1):
            piece = id_to_token.get(i, "<unk>")
            # una riga per ID, in ordine: indice = numero di riga
            f.write(piece + "\n")

    print(f"Vocab scritta in {out_path} con {max_id+1} token")


if __name__ == "__main__":
    base_dir = Path("granite-4.0-350m-BF16")
    model_path = "ibm-granite/granite-4.0-350m"
    out_path = base_dir / "vocab.txt"
    main(model_path, out_path)

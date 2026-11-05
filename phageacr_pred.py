import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


MAX_SEQ_LEN = 290
ESM_LAYER = 33
ESM_DIM = 1280


# ==========================================================================
# 1. Sequence standardization
# ==========================================================================
STANDARDIZE = {"B": "N", "Z": "Q", "U": "C", "O": "K", "J": "I", "X": ""}


def standardize(seq):
    seq = seq.replace(" ", "")
    for src, dst in STANDARDIZE.items():
        seq = seq.replace(src, dst)
    return seq


def pad_or_truncate(seq, max_len):
    if len(seq) > max_len:
        return seq[:max_len]
    return seq + "<pad>" * (max_len - len(seq))


# ==========================================================================
# 2. ESM-2 embeddings
# ==========================================================================
class Esm2Encoder:
    def __init__(self, batch_size=4, max_seq_len=MAX_SEQ_LEN, device="cpu"):
        import esm

        self.device = torch.device(device)
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.model = model.eval().to(self.device)
        self.batch_converter = alphabet.get_batch_converter()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

    @torch.no_grad()
    def embed(self, seqs):
        feats = []
        for start in range(0, len(seqs), self.batch_size):
            batch = [pad_or_truncate(s, self.max_seq_len) for s in seqs[start: start + self.batch_size]]
            data = [(str(i), s) for i, s in enumerate(batch)]
            _, _, tokens = self.batch_converter(data)
            reps = self.model(tokens.to(self.device), repr_layers=[ESM_LAYER],
                              return_contacts=False)["representations"][ESM_LAYER]
            for i, pstr in enumerate(batch):
                feats.append(reps[i, 1: len(pstr) + 1].mean(0).cpu().numpy())
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return np.stack(feats).astype("float32")


# ==========================================================================
# 3. MLP classifier
# ==========================================================================
class TorchMLPClassifier(nn.Module):
    def __init__(self, input_dim, num_layers=4, first_layer_output_dim=256,
                 dropout_prob=0.2, use_batch_norm=True):
        super().__init__()
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm

        self.input_layer = nn.Linear(input_dim, first_layer_output_dim)

        self.hidden = nn.ModuleList()
        hidden_size = first_layer_output_dim
        for _ in range(num_layers):
            self.hidden.append(nn.Linear(hidden_size, hidden_size // 2))
            hidden_size //= 2

        self.act = nn.ReLU()

        if use_batch_norm:
            self.batch_norm = nn.ModuleList()
            hidden_size = first_layer_output_dim
            for _ in range(num_layers):
                hidden_size //= 2
                self.batch_norm.append(nn.BatchNorm1d(hidden_size))

        self.dropout = nn.Dropout(p=dropout_prob)
        self.output_layers = nn.ModuleList([nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())])

    def forward(self, X):
        X = self.input_layer(X)
        for i in range(self.num_layers):
            X = self.hidden[i](X)
            X = self.act(X)
            if self.use_batch_norm:
                X = self.batch_norm[i](X)
            X = self.dropout(X)
        return [layer(X) for layer in self.output_layers]


def load_classifier(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k.removeprefix("model."): v for k, v in ck["state_dict"].items()}
    input_dim = sd["input_layer.weight"].shape[1]
    first_dim = sd["input_layer.weight"].shape[0]
    num_layers = sum(1 for k in sd if k.startswith("hidden.") and k.endswith(".weight"))
    use_bn = any(k.startswith("batch_norm.") for k in sd)
    model = TorchMLPClassifier(input_dim=input_dim, num_layers=num_layers,
                               first_layer_output_dim=first_dim, use_batch_norm=use_bn)
    model.load_state_dict(sd)
    return model.eval()


# ==========================================================================
# I/O
# ==========================================================================
def read_sequences(path):
    path = Path(path)
    ids, seqs = [], []
    if path.suffix.lower() in (".csv", ".tsv"):
        delim = "\t" if path.suffix.lower() == ".tsv" else ","
        with open(path, newline="") as f:
            for n, row in enumerate(csv.DictReader(f, delimiter=delim)):
                seqs.append(row["sequence"])
                ids.append(row.get("id") or row.get("ID") or f"seq_{n + 1}")
    else:
        cur = None
        for line in open(path):
            line = line.strip()
            if line.startswith(">"):
                ids.append(line[1:].split()[0]); seqs.append(""); cur = len(seqs) - 1
            elif line and cur is not None:
                seqs[cur] += line
    return ids, seqs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="FASTA, or CSV/TSV with a 'sequence' column")
    p.add_argument("-w", "--weights", required=True, help="trained .ckpt")
    p.add_argument("-o", "--output", default="predictions.csv")
    p.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--threshold", type=float, default=0.5)
    a = p.parse_args(argv)

    clf = load_classifier(a.weights)
    ids, seqs = read_sequences(a.input)
    seqs = [standardize(s) for s in seqs]
    print(f"{len(seqs)} sequences | embedding with ESM-2 650M on {a.device}", file=sys.stderr)

    enc = Esm2Encoder(batch_size=a.batch_size, max_seq_len=a.max_seq_len, device=a.device)
    X = enc.embed(seqs)

    with torch.no_grad():
        prob = clf(torch.tensor(X, dtype=torch.float32))[0].squeeze(-1).numpy()

    with open(a.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "acr_probability", "prediction"])
        for i, pr in zip(ids, np.atleast_1d(prob)):
            w.writerow([i, f"{pr:.4f}", "Acr" if pr >= a.threshold else "non-Acr"])
    print(f"wrote {a.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
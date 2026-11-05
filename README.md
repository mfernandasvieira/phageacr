# PhageAcr

PhageAcr predicts **anti-CRISPR (Acr)** proteins from amino-acid sequence. It encodes each protein with the ESM-2 protein language model and classifies it with a dense neural network.

---

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/mfernandasvieira/phageacr.git
cd phage_acr
pip install -r requirements.txt
```

<details>
<summary>Or install the dependencies directly</summary>

```bash
pip install fair-esm torch numpy
```
</details>

---

## Usage

```bash
python acr_predict.py proteins.fasta -w model.ckpt -o predictions.csv
```

### Input

A **FASTA** file:

```
>protein_1
MKFIKYLSTAHLNYMNIAVYENGSKIKARVENVVNGKSVGARDFDSTEQLESW
>protein_2
MMTISKTDIDCYLQTYVVIDPVSNGWQWGIDENGVGGALHHGRVEMVEG
```

…or a **CSV/TSV** with a `sequence` column (and an optional `id` column):

```csv
id,sequence
protein_1,MKFIKYLSTAHLNYMNIAVYENGSKIKARVENVVNGKSVGARDFDSTEQLESW
protein_2,MMTISKTDIDCYLQTYVVIDPVSNGWQWGIDENGVGGALHHGRVEMVEG
```

### Output

A CSV with one row per protein:

```csv
id,acr_probability,prediction
protein_1,0.9873,Acr
protein_2,0.0421,non-Acr
```

---

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `input` | — | FASTA, or CSV/TSV with a `sequence` column (positional) |
| `-w`, `--weights` | — | Path to the trained `.ckpt` (required) |
| `-o`, `--output` | `predictions.csv` | Output CSV path |
| `--max-seq-len` | `290` | Sequences are padded/truncated to this length |
| `--batch-size` | `4` | Sequences embedded per batch |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--threshold` | `0.5` | Probability cutoff for the `Acr` label |

Running on GPU:

```bash
python acr_predict.py proteins.fasta -w model.ckpt --device cuda --batch-size 16 -o predictions.csv
```


## License

Released under the MIT License — see [`LICENSE`](LICENSE).


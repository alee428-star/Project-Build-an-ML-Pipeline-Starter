import argparse
import pandas as pd
import scipy.stats
import tempfile
import subprocess
import os
import wandb


def kl_divergence(p, q):
    """Compute KL divergence between two distributions."""
    return scipy.stats.entropy(p, q)


def main(args):
    run = wandb.init(job_type="data_check")

    # Download latest cleaned sample
    csv_art = run.use_artifact(args.csv)
    csv_path = csv_art.file()

    # Download reference cleaned sample
    ref_art = run.use_artifact(args.ref)
    ref_path = ref_art.file()

    df = pd.read_csv(csv_path)
    df_ref = pd.read_csv(ref_path)

    # Run pytest on the data
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join("src", "data_check", "test_data.py")
        result = subprocess.run(
            ["pytest", "-q", test_file, "--disable-warnings", "--maxfail=1"],
            cwd=".",
            capture_output=True,
            text=True,
        )

        print(result.stdout)
        print(result.stderr)

        if result.returncode != 0:
            raise ValueError("Data tests failed")

    # Compute KL divergence on price distribution
    p = df["price"].value_counts(normalize=True).sort_index()
    q = df_ref["price"].value_counts(normalize=True).sort_index()

    # Align indexes
    p, q = p.align(q, fill_value=0)

    kl = kl_divergence(p, q)
    print(f"KL divergence: {kl}")

    if kl > args.kl_threshold:
        raise ValueError(f"KL divergence {kl} exceeds threshold {args.kl_threshold}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--ref", type=str, required=True)
    parser.add_argument("--kl_threshold", type=float, required=True)
    parser.add_argument("--min_price", type=float, required=True)
    parser.add_argument("--max_price", type=float, required=True)

    args = parser.parse_args()
    main(args)

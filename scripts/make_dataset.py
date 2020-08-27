import pickle
from collections import Counter
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    feat_type = "mel"
    exp_dir = Path("./training_data/exp_data/")  # base_out_dir from step2

    # ### Process ###
    feat_dir = exp_dir / feat_type
    feat_paths = sorted(feat_dir.glob("*.npy"))

    # feat_fns = [path.stem for path in feat_paths]

    dataset = [(path.stem, np.load(path).shape[-1]) for path in feat_paths]

    # Remove mels with different length.
    # Currently causes exception in training code:
    #
    #     RuntimeError: stack expects each tensor to be equal size,
    #     but got [80, 861] at entry 0 and [80, 27] at entry 3
    length_counter = Counter(length for _, length in dataset)
    dataset = [
        (id, length)
        for id, length in dataset
        if length == length_counter.most_common()[0][0]
    ]

    out_path = exp_dir / "dataset.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"Dataset length: {len(dataset)}")

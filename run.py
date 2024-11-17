import numpy as np
import argparse
from opmc import Trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--dataset', type=str, required=True, help="Path to the pickled dataset file")
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--device', str=int, default='cpu')
    parser.add_argument('--iterations', str=int, default=100)

    cfg = parser.parse_args()

    opmc_trainer = Trainer(cfg)

    all_vals = []
    for i in range(cfg.seeds):
        np.random.seed(i)
        print(f"\nSeed: {i}")
        all_vals.append(opmc_trainer.train_eval())

    all_vals = np.array(all_vals)
    print(f"ACC:{all_vals[:, 0]}")
    print(f"ACC mean:{all_vals[:, 0].mean():0.4f}, ACC std: {all_vals[:, 0].std():0.4f}")

    print(f"ARI:{all_vals[:, 2]}")
    print(f"ARI mean:{all_vals[:, 2].mean():0.4f}, ARI std: {all_vals[:, 2].std():0.4f}")

    print(f"NMI:{all_vals[:, 1]}")
    print(f"NMI mean:{all_vals[:, 1].mean():0.4f}, NMI std: {all_vals[:, 1].std():0.4f}")

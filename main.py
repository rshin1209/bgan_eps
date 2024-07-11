import os
import importlib
import torch
import joblib
import argparse
import numpy as np
import mdtraj as md
from bgan import BGANEPS
from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(42)
torch.cuda.manual_seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--reaction", type=str, default="", help="Name of the reaction file without format tag")
    parser.add_argument(
        "--ensemble", type=int, default=10, help="Number of structural ensembles for entropic path sampling"
    )
    parser.add_argument(
        "--bondmax", type=float, default=2.897, help="Maximum bond length (i.e., bond length in the optimized TS structure)"
    )
    parser.add_argument(
        "--bondmin", type=float, default=1.700, help="Minimum bond length (i.e., bond formation criterion)"
    )
    parser.add_argument(
        "--temperature", type=float, default=298.15, help="Temperature in configurational entropy calculation"
    )
    parser.add_argument(
        "--eps_type",
        type=str,
        default="average",
        help="Type of entropic path sampling: average or max (average recommended)",
    )

    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for BGAN training (64 recommended)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs for BGAN training (50 recommended)"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate of Adam Optimizer in BGAN training (1e-4 recommended)")
    parser.add_argument("--beta1", type=float, default=0.5, help="Momentum1 for Adam Optimizer (0.5 recommended)")
    parser.add_argument("--beta2", type=float, default=0.999, help="Momentum2 for Adam Optimizer (0.999 recommended)")
    parser.add_argument(
        "--loop", type=int, default=20, help="Number of BGAN-EPS rounds (5-20 recommended based on available computation resources)"
    )

    args = parser.parse_args()
    output_dir = f"./log/{args.reaction}"

    X = np.load(f"{output_dir}/dof.npy")
    dim = X.shape[1]
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X)
    X = scaler.transform(X)

    for _ in range(args.loop):
        BGANEPS(output_dir).run(args, X, dim, scaler, _)

    ep = np.load(f"{output_dir}/bgan1/eps.npy")
    for i in range(2, args.loop + 1):
        ep = np.vstack((ep, np.load(f"{output_dir}/bgan{i}/eps.npy")))

    np.save(f"{output_dir}/{args.reaction}_eps.npy", ep)

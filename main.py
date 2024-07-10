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
    parser.add_argument("--reaction", type=str, default="")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="maximum training epoches"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)

    parser.add_argument(
        "--ensemble", type=int, default=10, help="number of structural ensembles"
    )
    parser.add_argument(
        "--bondmax", type=float, default=2.897, help="reaction coordinate max"
    )
    parser.add_argument(
        "--bondmin", type=float, default=1.700, help="reaction coordinate min"
    )
    parser.add_argument(
        "--temperature", type=float, default=298.15, help="temperature in Kevlin"
    )
    parser.add_argument(
        "--loop", type=int, default=20, help="number of BGAN-EPS rounds"
    )
    parser.add_argument(
        "--eps_type",
        type=str,
        default="average",
        help="Type of EPS: Average or Maximal Entropy Approach",
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

#!/usr/bin/env python3
# linear_regression_gd.py
# Run: python linear_regression_gd.py --lr 0.05 --iters 400 --n 200 --seed 42 --save
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def predict(x, w, b):
    return w * x + b

def mse_loss(x, y, w, b):
    y_hat = predict(x, w, b)
    return 0.5 * np.mean((y_hat - y) ** 2)  # 0.5 makes gradients cleaner

def gradients(x, y, w, b):
    """
    ∂J/∂w = (1/n) * Σ (y_hat - y) * x
    ∂J/∂b = (1/n) * Σ (y_hat - y)
    """
    y_hat = predict(x, w, b)
    err = y_hat - y
    dw = np.mean(err * x)
    db = np.mean(err)
    return dw, db

def make_synthetic(n, seed, true_w=2.5, true_b=-1.0, x_low=-3.0, x_high=3.0, noise_std=0.7):
    rng = np.random.default_rng(seed)
    X = rng.uniform(x_low, x_high, size=n)
    noise = rng.normal(0, noise_std, size=n)
    y = true_w * X + true_b + noise
    return X, y, true_w, true_b

def train_gd(X, y, lr=0.05, iters=400, w0=0.0, b0=0.0):
    w, b = w0, b0
    loss_hist = []
    for _ in range(iters):
        dw, db = gradients(X, y, w, b)
        w -= lr * dw
        b -= lr * db
        loss_hist.append(mse_loss(X, y, w, b))
    return w, b, loss_hist

def plot_loss(loss_hist, outdir=None, show=True):
    plt.figure(figsize=(5, 3))
    plt.plot(loss_hist)
    plt.xlabel("Iteration")
    plt.ylabel("Loss (MSE)")
    plt.title("Training Loss vs Iterations")
    plt.tight_layout()
    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(outdir) / "loss_curve.png", dpi=160)
    if show:
        plt.show()
    plt.close()

def plot_fit(X, y, w, b, true_w, true_b, outdir=None, show=True):
    xs = np.linspace(X.min(), X.max(), 200)
    plt.figure(figsize=(5, 3))
    plt.scatter(X, y, s=12, alpha=0.6, label="data")
    plt.plot(xs, true_w * xs + true_b, linestyle="--", label="true line")
    plt.plot(xs, w * xs + b, label="learned line")
    plt.legend()
    plt.tight_layout()
    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(outdir) / "fit.png", dpi=160)
    if show:
        plt.show()
    plt.close()

def parse_args():
    p = argparse.ArgumentParser(description="Linear regression via gradient descent (from scratch).")
    p.add_argument("--lr", type=float, default=0.05, help="Learning rate.")
    p.add_argument("--iters", type=int, default=400, help="Number of GD iterations.")
    p.add_argument("--n", type=int, default=200, help="Number of synthetic samples.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--save", action="store_true", help="Save plots to ./outputs/.")
    p.add_argument("--no-show", action="store_true", help="Do not display plots (useful on headless).")
    return p.parse_args()

def main():
    args = parse_args()
    X, y, true_w, true_b = make_synthetic(n=args.n, seed=args.seed)
    w, b, loss_hist = train_gd(X, y, lr=args.lr, iters=args.iters)

    print(f"Learned w={w:.3f}, b={b:.3f} (true w={true_w}, b={true_b})")
    outdir = "outputs" if args.save else None
    show = not args.no_show

    plot_loss(loss_hist, outdir=outdir, show=show)
    plot_fit(X, y, w, b, true_w, true_b, outdir=outdir, show=show)

if __name__ == "__main__":
    main()

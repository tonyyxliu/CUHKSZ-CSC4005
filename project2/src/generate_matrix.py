#
# Created by Sergei Kudria on 2024Fall
#
# Modified by Liu Yuxuan on 2025Fall
#
# Script to generate dense matrices
#

import sys
import random
import argparse


def generate_matrix(N, filename):
    """
    Randomly generate N * N square matrix in double, and save to .txt file
    """
    with open(filename, "w") as f:
        f.write(f"{N} {N}\n")

        for i in range(N):
            row = [random.uniform(-100, 100) for _ in range(N)]
            # 8 digits precision double
            f.write(" ".join(f"{x:.4f}" for x in row) + "\n")

    print(f"Randomly generate {N}x{N} matrix of type double to file: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Random N * N double matrix generator")
    parser.add_argument("N", type=int, help="Matrix dimension")
    parser.add_argument("filename", type=str, help="output filename")

    args = parser.parse_args()

    if args.N <= 0:
        sys.exit(1)

    generate_matrix(args.N, args.filename)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_matrix.py N filename")
        sys.exit(1)
    main()

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import re


def main(num_epochs, num_seeds, saving_dir, results_training_dir):
    """
    A utility for averaging and visualizing training results based on different
    dataset sizes.

    Args:
        num_epochs (int): Number of epochs.
        num_seeds (int): Number of seeds.
        saving_dir (Path): Directory to save the generated figures.
        results_training_dir (Path): Directory containing the results of the
        training.

    Raises:
        ValueError: If the provided results_training_dir does not contain the
        required patterns.

    Note:
        Assumes that the results_training_dir follows a specific structure with
        placeholders 'seed=x' and 'size_dataset=x'.
    """
    list_train_losses = []
    list_test_losses = []
    list_train_top1 = []
    list_test_top1 = []
    list_pathnorm1 = []
    list_pathnorm2 = []
    list_pathnorm4 = []

    fraction_size_dataset = {
        39636: "1/32",
        79272: "1/16",
        158544: "1/8",
        317089: "1/4",
        634178: "1/2",
    }

    print(f"=> Averaging results from {num_seeds} seeds.")
    for size_dataset in [39636, 79272, 158544, 317089, 634178]:
        for seed in range(num_seeds):
            results_training_path = (
                results_training_dir
                / f"rank={args.rank}"
                / "csv"
                / "results.csv"
            )
            # look for a pattern seed=x and replace x by the new seed
            pattern = re.compile(r"seed=\d+")
            results_training_path = Path(
                str(results_training_path).replace(
                    pattern.search(str(results_training_path)).group(),
                    f"seed={seed}",
                )
            )
            # look for a pattern size_dataset=x and replace x by the new size_dataset
            pattern = re.compile(r"size_dataset=\d+")
            results_training_path = Path(
                str(results_training_path).replace(
                    pattern.search(str(results_training_path)).group(),
                    f"size_dataset={size_dataset}",
                )
            )
            df = pd.read_csv(results_training_path)

            epochs = df["epoch"].values[:num_epochs]
            if seed == 0:
                train_losses = df["train/loss"].values[:num_epochs]
                test_losses = df["test/loss"].values[:num_epochs]
                train_top1 = df["train/acc1"].values[:num_epochs]
                test_top1 = df["test/acc1"].values[:num_epochs]

                pathnorm1 = df["pathnorm1"].values[:num_epochs]
                pathnorm2 = df["pathnorm2"].values[:num_epochs]
                pathnorm4 = df["pathnorm4"].values[:num_epochs]
            else:
                train_losses += df["train/loss"].values[:num_epochs]
                test_losses += df["test/loss"].values[:num_epochs]
                train_top1 += df["train/acc1"].values[:num_epochs]
                test_top1 += df["test/acc1"].values[:num_epochs]

                pathnorm1 += df["pathnorm1"].values[:num_epochs]
                pathnorm2 += df["pathnorm2"].values[:num_epochs]
                pathnorm4 += df["pathnorm4"].values[:num_epochs]
        train_losses /= num_seeds
        test_losses /= num_seeds
        train_top1 /= num_seeds
        test_top1 /= num_seeds
        pathnorm1 /= num_seeds
        pathnorm2 /= num_seeds
        pathnorm4 /= num_seeds

        list_train_losses.append(train_losses)
        list_test_losses.append(test_losses)
        list_train_top1.append(train_top1)
        list_test_top1.append(test_top1)
        list_pathnorm1.append(pathnorm1)
        list_pathnorm2.append(pathnorm2)
        list_pathnorm4.append(pathnorm4)

    print(f"=> Saving plots in {args.saving_dir}")

    colormap = plt.get_cmap("hot")
    colors = [
        colormap(i / len(list_train_losses))
        for i in range(len(list_train_losses))
    ]

    # plot top 1 generalization error
    for i, size_dataset in enumerate([39636, 79272, 158544, 317089, 634178]):
        # plt.plot(
        #     epochs,
        #     100 - list_train_top1[i],
        #     label=f"Train {fraction_size_dataset[size_dataset]}",
        #     color=colors[i],
        # )
        # plt.plot(
        #     epochs,
        #     100 - list_test_top1[i],
        #     label=f"Test {fraction_size_dataset[size_dataset]}",
        #     color=colors[i],
        # )
        gene_top_1 = (100 - list_test_top1[i]) - (100 - list_train_top1[i])
        plt.plot(
            epochs,
            gene_top_1,
            label=f"{fraction_size_dataset[size_dataset]}",
            color=colors[i],
        )
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Generalization error for top-1 accuracy (%)")
    plt.savefig(saving_dir / "top1.pdf")
    plt.close()

    # plot cross-entropy generalization error
    for i, size_dataset in enumerate([39636, 79272, 158544, 317089, 634178]):
        # plt.plot(
        #     epochs,
        #     list_train_losses[i],
        #     label=f"Train {fraction_size_dataset[size_dataset]}",
        #     color=colors[i],
        # )
        # plt.plot(
        #     epochs,
        #     list_test_losses[i],
        #     label=f"Test {fraction_size_dataset[size_dataset]}",
        #     color=colors[i],
        # )
        generalization_error_CE = list_test_losses[i] - list_train_losses[i]
        plt.plot(
            epochs,
            generalization_error_CE,
            label=f"{fraction_size_dataset[size_dataset]}",
            color=colors[i],
        )

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Generalization error for the cross-entropy loss")
    plt.savefig(saving_dir / "cross_entropy.pdf")
    plt.close()

    # plot L1 path-norm
    for i, size_dataset in enumerate([39636, 79272, 158544, 317089, 634178]):
        plt.plot(
            epochs,
            list_pathnorm1[i],
            label=f"{fraction_size_dataset[size_dataset]}",
            color=colors[i],
        )

    plt.legend(loc="best")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("L1 path-norm")
    plt.savefig(saving_dir / "L1_path_norm.pdf")
    plt.close()

    # plot L1, L2, L4 path-norms
    for i, size_dataset in enumerate([39636, 79272, 158544, 317089, 634178]):
        plt.plot(
            epochs,
            list_pathnorm1[i],
            label=f"L1 {fraction_size_dataset[size_dataset]}",
            color=colors[i],
        )
        plt.plot(
            epochs,
            list_pathnorm2[i],
            label=f"L2 {fraction_size_dataset[size_dataset]}",
            color=colors[i],
        )
        plt.plot(
            epochs,
            list_pathnorm4[i],
            label=f"L4 {fraction_size_dataset[size_dataset]}",
            color=colors[i],
        )
    plt.legend()
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Path-norm")
    plt.savefig(saving_dir / "path_norms.pdf")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-epochs", type=int, default=90)
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=3,
        help="Will plot the average results of 4_train_increasing_dataset.sh over all integer seeds in [0, num_seeds).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=-1,
        help="node rank for distributed training",
    )
    parser.add_argument(
        "--saving-dir",
        type=Path,
        default=Path("results/5_plot_increasing_dataset/"),
    )
    parser.add_argument(
        "--results-training-dir",
        type=Path,
        default=None,
        help="Saving directory used in script 4_train_increasing_dataset.sh for an arbitrary seed: where the results of training have been saved.",
    )
    args = parser.parse_args()

    if args.results_training_dir is None:
        raise ValueError("Please provide --results-training-dir.")
    pattern = re.compile(r"seed=\d+")
    if not pattern.search(str(args.results_training_dir)):
        raise ValueError(
            "--results-training-dir must be a directory that contains the string 'seed=x', where x is an arbitrary integer. x will be replaced with all integers in [0, num_seeds)."
        )

    pattern = re.compile(r"size_dataset=\d+")
    if not pattern.search(str(args.results_training_dir)):
        raise ValueError(
            "--results-training-dir must be a directory that contains the string 'size_dataset=x', where x is an arbitrary integer. x will be replaced with all sizes in [39636, 79272, 158544, 317089, 634178]"
        )

    args.saving_dir.mkdir(parents=True, exist_ok=True)

    main(
        args.num_epochs,
        args.num_seeds,
        args.saving_dir,
        args.results_training_dir,
    )

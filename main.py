from argparse import ArgumentParser, Namespace

import pandas as pd

from automl import AutoML


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Module to find a model that works the best for the data provided to the program."
    )
    parser.add_argument("--train-data", type=str, help="Path to the training data.")
    parser.add_argument(
        "--target", type=str, help="Name of the target column in the training data."
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Type of the task. Either classification or regression.",
        default="classification",
        choices=["classification", "regression"],
    )
    parser.add_argument(
        "--log",
        type=bool,
        default=False,
        help="If provided, the script will create logs at debug level. Only use the flag if working on the library "
        "improvements.",
    )
    parser.add_argument(
        "--fast-mode",
        type=bool,
        default=False,
        help="If provided, the script will not use StratifiedKFold for cross validation. It will use "
        "train-test-split instead.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_data = pd.read_csv(args.train_data)
    automl = AutoML(
        train_data=train_data,
        target=args.target,
        task=args.task,
        log=args.log,
        fast_mode=args.fast_mode,
    )
    automl.fit()


if __name__ == "__main__":
    main()

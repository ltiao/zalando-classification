"""Console script for zalando_classification."""
import sys
import click


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from zalando_classification.datasets import load_zalando
from zalando_classification.utils import maybe_load_model, build_callbacks


# Sensible defaults
EPOCHS = 50
BATCH_SIZE = 64

OPTIMIZER = "rmsprop"

BATCH_NORM = False
L1_FACTOR = 0.0
L2_FACTOR = 1e-5

SPLIT_METHOD = "shuffle"
N_SPLITS = 1
RESUME_FROM_EPOCH = 0

TEST_SIZE = 0.4
SEED = None

CHECKPOINT_DIR = "models/"
CHECKPOINT_PERIOD = 10

SUMMARY_DIR = "logs/"


@click.command()
@click.argument("name")
@click.option("--optimizer", default=OPTIMIZER)
@click.option("-e", "--epochs", default=EPOCHS, type=int,
              help="Number of epochs.")
@click.option("-b", "--batch-size", default=BATCH_SIZE, type=int,
              help="Batch size.")
@click.option('--evaluate-only', is_flag=True,
              help="Skip model fitting. Only evaluate model.")
@click.option("--resume-from-epoch", default=RESUME_FROM_EPOCH, type=int,
              help="Epoch at which to resume a previous training run")
@click.option("--l1-factor", default=L1_FACTOR, type=float,
              help="L1 regularization factor.")
@click.option("--l2-factor", default=L2_FACTOR, type=float,
              help="L2 regularization factor.")
@click.option("--batch-norm/--no-batch-norm", default=BATCH_NORM,
              help="Use Batch Normalization (after activation).")
@click.option('--standardize', is_flag=True)
@click.option("--split-method", default=SPLIT_METHOD,
              type=click.Choice(["kfold", "shuffle"]),
              help="Method for generating train/test dataset splits.")
@click.option("--n-splits", default=N_SPLITS, type=int,
              help="Number of train/test dataset splits.")
@click.option("--test-size", default=TEST_SIZE, type=float,
              help="Test set size (for shuffle split method only).")
@click.option("--checkpoint-dir", default=CHECKPOINT_DIR,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Model checkpoint directory.")
@click.option("--checkpoint-period", default=CHECKPOINT_PERIOD, type=int,
              help="Interval (number of epochs) between checkpoints.")
@click.option("--summary-dir", default=SUMMARY_DIR,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Summary directory.")
@click.option("-s", "--seed", default=SEED, type=int, help="Random seed")
def main(name, optimizer, epochs, batch_size, evaluate_only, resume_from_epoch,
         l1_factor, l2_factor, batch_norm, standardize, split_method, n_splits,
         test_size, checkpoint_dir, checkpoint_period, summary_dir, seed):

    # Data loading
    X, y = load_zalando()

    if split_method == "kfold":

        splitter = StratifiedKFold(n_splits=n_splits,
                                   shuffle=True,
                                   random_state=seed)
    else:

        splitter = StratifiedShuffleSplit(n_splits=n_splits,
                                          test_size=test_size,
                                          random_state=seed)

    for split_num, (train_ind, test_ind) in enumerate(splitter.split(X, y)):

        X_train, y_train = X[train_ind], y[train_ind]
        X_test, y_test = X[test_ind], y[test_ind]

        # Data pre-processing
        if standardize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Model specification
        model, initial_epoch = maybe_load_model(name, split_num, checkpoint_dir,
                                                resume_from_epoch, batch_norm,
                                                l1_factor, l2_factor,
                                                optimizer)

        # Model fitting
        if not evaluate_only:

            callbacks = build_callbacks(name, split_num, summary_dir,
                                        checkpoint_dir, checkpoint_period)

            hist = model.fit(X_train, y_train, batch_size=batch_size,
                             epochs=epochs, validation_data=(X_test, y_test),
                             shuffle=True, initial_epoch=initial_epoch,
                             callbacks=callbacks)

        # Model evaluation
        loss_test, acc_test = model.evaluate(X_test, y_test)

        click.secho(f"[Split {split_num:d}] test accuracy: {acc_test:.3f}, "
                    f"test loss {loss_test:.3f}", fg='green')

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover

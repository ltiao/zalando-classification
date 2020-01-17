"""Utils module."""
import click
import os.path

from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard

from zalando_classification.models import build_model


def maybe_load_model(name, split_num, checkpoint_dir, resume_from_epoch,
                     batch_norm, l1_factor, l2_factor, optimizer):
    """
    Attempt to load the specified model (including architecture, weights, and
    even optimizer states). If this is not possible, build a new model from
    scratch.
    """
    basename = get_basename(name, split_num)
    model_filename_fmt = get_model_filename_fmt(basename)
    model_filename = model_filename_fmt.format(epoch=resume_from_epoch)

    checkpoint_path = os.path.join(checkpoint_dir, model_filename)

    if resume_from_epoch > 0 and os.path.isfile(checkpoint_path):

        click.secho(f"Found model checkpoint '{checkpoint_path}'. "
                    f"Resuming from epoch {resume_from_epoch}.", fg='green')

        model = load_model(checkpoint_path)

        initial_epoch = resume_from_epoch

    else:

        click.secho(f"Could not load model checkpoint '{checkpoint_path}' "
                    "or `resume_from_epoch == 0`. Building new model.",
                    fg='yellow')

        model = build_model(output_dim=1, batch_norm=batch_norm,
                            kernel_regularizer=l1_l2(l1_factor, l2_factor))
        # optimizer = Adam(beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])

        initial_epoch = 0

    return model, initial_epoch


def build_callbacks(name, split_num, summary_dir, checkpoint_dir,
                    checkpoint_period):

    basename = get_basename(name, split_num)
    model_filename_fmt = get_model_filename_fmt(basename)

    tensorboard_path = os.path.join(summary_dir, basename)
    csv_path = os.path.join(summary_dir, f"{basename}.csv")
    checkpoint_path = os.path.join(checkpoint_dir, model_filename_fmt)

    callbacks = []
    callbacks.append(TensorBoard(tensorboard_path, profile_batch=0))
    callbacks.append(CSVLogger(csv_path, append=True))
    callbacks.append(ModelCheckpoint(checkpoint_path, period=checkpoint_period))

    return callbacks


def get_basename(name, split_num):

    return f"{name}.split{split_num:d}"


def get_model_filename_fmt(basename):

    return f"{basename}.{{epoch:02d}}.h5"

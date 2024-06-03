import re,random
from pandas import read_csv
import tensorflow as tf


def load_translation(data=None, filepath=None, sep=";", val_rate=0.1):
    # Ensure either data or filepath is provided
    assert data is not None or filepath is not None, "Either data or filepath must be provided."
    
    # Load data from provided DataFrame or from a CSV file
    if data is not None:
        df = data
    elif filepath is not None:
        df = read_csv(filepath, sep=sep)

    df_list = list(df.itertuples(index=False, name=None))
    random.shuffle(df_list)
    dataset_size = len(df_list)
    val_size = int(dataset_size * val_rate)
    train_size = dataset_size - val_size
    dataset = tf.data.Dataset.from_tensor_slices(df_list)
    dataset = dataset.map(lambda x: tf.unstack(x))

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    return {"train": train_dataset, "validation": val_dataset}


def add_start_end(ragged, reserved_tokens):
    START = tf.argmax(tf.constant(reserved_tokens) == "[START]")
    END = tf.argmax(tf.constant(reserved_tokens) == "[END]")

    count = ragged.bounding_shape()[0]
    starts = tf.fill([count, 1], START)
    ends = tf.fill([count, 1], END)
    return tf.concat([starts, ragged, ends], axis=1)


def cleanup_text(reserved_tokens, token_txt):
    # Drop the reserved tokens, except for "[UNK]".
    bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]
    bad_token_re = "|".join(bad_tokens)

    bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
    result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

    # Join them into strings.
    result = tf.strings.reduce_join(result, separator=' ', axis=-1)

    return result

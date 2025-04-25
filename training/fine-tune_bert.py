import os
import argparse
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, create_optimizer
from sklearn.model_selection import train_test_split

def load_data(csv_path, test_size=0.1, random_state=42):
    """
    Load CSV and split into train/validation DataFrames.
    Assumes a 'review' column and a 'sentiment' column with values 'positive'/'negative'.
    Converts sentiment to integer labels.
    """
    df = pd.read_csv(csv_path)
    # map string labels to integers
    label_map = {'negative': 0, 'positive': 1}
    df['sentiment'] = df['sentiment'].map(label_map)
    if df['sentiment'].isnull().any():
        raise ValueError("Found sentiment labels outside of 'positive'/'negative' in CSV.")
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df.sentiment
    )
    return train_df, val_df


def encode_dataset(df, tokenizer, max_len):
    """
    Tokenize texts and prepare a tf.data.Dataset with integer labels.
    """
    encodings = tokenizer(
        df.review.tolist(),
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='tf'
    )
    labels = tf.convert_to_tensor(df.sentiment.values, dtype=tf.int32)
    return tf.data.Dataset.from_tensor_slices((dict(encodings), labels))


def configure_gpu(mixed_precision=False):
    """
    Enable GPU memory growth and optional mixed precision.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
            if mixed_precision:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                print("Mixed precision enabled.")
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")
    else:
        print("No GPU detected, running on CPU.")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT for IMDB Sentiment Classification"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of epochs to train (default: 3)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="models/bert_finetuned",
        help="Directory to save the fine-tuned model & tokenizer"
    )
    args = parser.parse_args()

    # Paths and hyperparameters
    PROJECT_ROOT = os.path.dirname(__file__)
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'IMDB_Dataset.csv')
    BERT_MODEL = 'bert-base-uncased'
    MAX_LEN = 128
    BATCH_SIZE = 16  # adjust down if OOM on 6GB GPU
    LEARNING_RATE = 2e-5
    TEST_SIZE = 0.1

    # GPU and data
    configure_gpu(mixed_precision=True)
    train_df, val_df = load_data(DATA_PATH, test_size=TEST_SIZE)
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    train_ds = encode_dataset(train_df, tokenizer, MAX_LEN)
    val_ds = encode_dataset(val_df, tokenizer, MAX_LEN)

    train_ds = train_ds.shuffle(10_000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Model and optimizer
    model = TFBertForSequenceClassification.from_pretrained(
        BERT_MODEL,
        num_labels=2
    )
    steps_per_epoch = len(train_df) // BATCH_SIZE
    total_steps = steps_per_epoch * args.epochs
    optimizer, lr_schedule = create_optimizer(
        init_lr=LEARNING_RATE,
        num_warmup_steps=int(0.1 * total_steps),
        num_train_steps=total_steps
    )
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    # Training
    class TimeLogger(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.start_time = tf.timestamp()
        def on_epoch_end(self, epoch, logs=None):
            end_time = tf.timestamp()
            duration = (end_time - self.start_time).numpy() / 60
            print(f"⚡️ Epoch {epoch+1} took {duration:.2f} min")

    callbacks = [TimeLogger()]
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model and tokenizer saved to {args.output_dir}")

if __name__ == '__main__':
    main()

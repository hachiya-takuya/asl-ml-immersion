"""here it is"""

import datetime
import os

import keras_nlp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Embedding,
    Input,
    Layer,
    LayerNormalization,
    MultiHeadAttention,
)


class TimestampedModelCheckpoint(tf.keras.callbacks.Callback):
    """timestamp check point call back"""

    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.saved_models = []

    def on_epoch_end(self, epoch, logs=None):
        """on epoch end"""
        if (epoch + 1) % 8 == 0:
            _ = logs
            timestamp = datetime.datetime.now().isoformat(timespec="seconds")
            safe_timestamp = timestamp.replace(":", "_")
            filename = f"model_{safe_timestamp}_epoch{epoch}"
            filepath = os.path.join(self.save_dir, filename)
            self.model.save(filepath)
            print(f">>> Saved model to {filepath}")
            self.saved_models.append(filepath)

            while len(self.saved_models) > 16:
                to_delete = self.saved_models.pop(0)
                print(f">>> Deleting old model: {to_delete}")
                tf.io.gfile.rmtree(to_delete)


class TransformerBlock(Layer):
    """transformer block"""

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                Dense(ff_dim, activation="relu"),
                Dense(embed_dim, activation="relu"),
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs):
        """call"""
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(Layer):
    """class"""

    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.maxlen = maxlen
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        """call"""
        seq_len = tf.shape(x)[-1]
        pad_len = self.maxlen - seq_len

        x = tf.cond(
            pad_len > 0,
            lambda: tf.pad(x, paddings=[[0, 0], [0, pad_len]], constant_values=0),
            lambda: x[:, :self.maxlen]
        )
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def masked_sparse_categorical_crossentropy(pad_token_id):
    """masked sparse categorical crossentropy"""
    def _loss(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, pad_token_id), tf.float32)
        loss_ = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
        loss_ = loss_ * mask
        _loss.__name__ = "masked_sparse_categorical_crossentropy"
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)
    return _loss


class Transformer:
    """transformer"""

    @classmethod
    def load_model(
        cls,
        model_path,
        tokenizer,
        maxlen=128,
        vocab_size=4096
    ):
        """
        保存されたモデルから復元する。
        - model_path: 保存先のパス
        - tokenizer: 既存のtokenizerインスタンス
        - その他: モデル構成と一致させる必要あり
        """
        pad_token_id = tokenizer.token_to_id("[PAD]")
        custom_objects = {
            'masked_sparse_categorical_crossentropy': masked_sparse_categorical_crossentropy(pad_token_id),
            'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
            'TransformerBlock': TransformerBlock,
        }
        model = keras.models.load_model(model_path, custom_objects=custom_objects)

        instance = cls(
            maxlen=maxlen,
            vocab_size=vocab_size,
            tokenizer=tokenizer,
        )
        instance.model = model
        return instance

    def __init__(
        self,
        embed_dim: int = 32,  # Embedding size for each token
        num_heads: int = 2,  # Number of attention heads
        ff_dim: int = 32,  # Hidden layer size in feed forward network inside transformer
        maxlen: int = 2048,
        loop_n: int = 12,
        vocab_size: int = 32000,
        tokenizer=None,
    ):
        self.history = None
        self.maxlen = maxlen
        self.pad_token_id = 0  # default
        inputs = Input(shape=(maxlen,))
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        x = self.embedding_layer(inputs)
        for _ in range(loop_n):
            transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
            x = transformer_block(x)

        x = Dropout(0.1)(x)
        x = Dense(ff_dim, activation="relu")(x)
        x = Dropout(0.1)(x)
        outputs = Dense(vocab_size, activation="softmax")(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
            loss=masked_sparse_categorical_crossentropy(pad_token_id=self.pad_token_id),
            metrics=[keras_nlp.metrics.Perplexity(from_logits=False, mask_token_id=0)],
            # metrics=["accuracy"],
        )
        self.tokenizer = tokenizer
        if self.tokenizer:
            self.start_packer = keras_nlp.layers.StartEndPacker(
                sequence_length=self.maxlen,
                start_value=tokenizer.token_to_id("[BOS]"),
                end_value=tokenizer.token_to_id("[EOS]"),
                pad_value=tokenizer.token_to_id("[PAD]"),
            )
            self.pad_token_id = self.tokenizer.token_to_id("[PAD]")

    def train_tokenizer(self, data, vocab_size=4096):
        """train_tokenizer"""
        vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
            data,
            vocabulary_size=vocab_size,
            lowercase=True,
            reserved_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
        )
        tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
            vocabulary=vocab,
            sequence_length=self.maxlen,
            lowercase=True,
        )
        self.model.tokenizer = tokenizer
        self.start_packer = keras_nlp.layers.StartEndPacker(
            sequence_length=self.maxlen,
            start_value=tokenizer.token_to_id("[BOS]"),
            end_value=tokenizer.token_to_id("[EOS]"),
            pad_value=tokenizer.token_to_id("[PAD]"),
        )
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")

    def train(
            self,
            *,
            train_dataset,
            validation_data,
            steps_per_epoch,
            epochs,
            lr: float = None
    ):
        """ train """
        if lr is not None:
            keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        self.history = self.model.fit(
            train_dataset,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=[TimestampedModelCheckpoint(save_dir="./variables")],
        )

    def generate(self, text: str, p: float = None):
        """generate"""
        input_tokens = self.tokenizer([text])
        packed_tokens = self.start_packer(input_tokens)
        token_length = tf.where(packed_tokens != 0)[-1, 1]
        initial_sequence_length = token_length + 1
        gen_ittr = self._generate_step(
            tokens=packed_tokens,
            p=p,
            start_index=int(initial_sequence_length.numpy()),
        )
        generated_text_parts = [text]
        for word in gen_ittr:
            generated_text_parts.append(word)
            print(word, end=" ")

        return " ".join(generated_text_parts)

    def _generate_step(self, tokens, p=None, start_index=1):
        tokens = tokens.numpy()
        for i in range(start_index, self.maxlen):
            sampled_token = len(self.tokenizer.vocabulary)
            while sampled_token > len(self.tokenizer.vocabulary) - 1:
                logits = self.model.predict([tokens], verbose=0)[:, i - 1, :]
                logits = tf.constant(logits)
                sampled_token = self.top_p_sample(logits[0], p)

            tokens[0][i] = sampled_token
            next_word = self.tokenizer.detokenize([sampled_token]).numpy().decode()
            yield next_word
            if sampled_token == self.tokenizer.token_to_id("[EOS]"):
                return

    @staticmethod
    def top_p_sample(logits, p=None):
        """top sample"""
        if p is None:
            p = 0.02
        probs = tf.nn.softmax(logits)
        sorted_probs, sorted_indices = tf.sort(probs, direction="DESCENDING"), tf.argsort(probs, direction="DESCENDING")
        cumulative_probs = tf.cumsum(sorted_probs)

        cutoff_index = tf.reduce_min(tf.where(cumulative_probs > p))
        cutoff_index = tf.maximum(cutoff_index, 1)
        top_p_indices = sorted_indices[:cutoff_index]
        top_p_logits = tf.gather(logits, top_p_indices)
        sampled_relative = tf.random.categorical([top_p_logits], num_samples=1)[0, 0]
        sampled_token = top_p_indices[sampled_relative]

        return sampled_token


def _build_token_dataset(
    BATCH_SIZE=64,
    MIN_TRAINING_SEQ_LEN=92
):
    """
    for create dataset to train tokenizer
    if you want to train tokenizer local,

    ds = _build_token_dataset()
    Run Transformer.train_tokenizer(ds)
    """
    # Data

    keras.utils.get_file(
        origin="https://storage.googleapis.com/asl-public/text/data/simplebooks.zip",
        extract=True,
    )
    data_dir = os.path.expanduser("./data/")

    # Load simplebooks-92 train set and filter out short lines using MIN_TRAINING_SEQ_LEN
    raw_train_ds = (
        tf.data.TextLineDataset(data_dir + "simplebooks-92-raw/train.txt")
        .filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
        .batch(BATCH_SIZE)
        .shuffle(buffer_size=256)
    )

    # Load simplebooks-92 validation set and filter out short lines using MIN_TRAINING_SEQ_LEN
    raw_val_ds = (
        tf.data.TextLineDataset(data_dir + "simplebooks-92-raw/valid.txt")
        .filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
        .batch(BATCH_SIZE)
    )
    return raw_train_ds, raw_val_ds


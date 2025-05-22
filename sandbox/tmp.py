"""here it is"""
import numpy as np

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Layer,
    Input,
    MultiHeadAttention,
    Dense,
    LayerNormalization,
    Dropout,
    Embedding,
)

import keras_nlp


class TransformerBlock(Layer):
    """transformer block"""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim), ]
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
        """call """
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


class Transformer:
    """transformer"""
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
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.tokenizer = tokenizer
        if self.tokenizer:
            self.start_packer = keras_nlp.layers.StartEndPacker(
                sequence_length=self.maxlen,
                start_value=tokenizer.token_to_id("[BOS]"),
            )

    def train_tokenizer(self, data, vocab_size=4096):
        """train_tokenizer"""
        vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
            data,
            vocabulary_size=vocab_size,
            lowercase=True,
            reserved_tokens=["[PAD]", "[UNK]", "[BOS]"],
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
        )

    def train(
            self,
            *,
            train_dataset,
            validation_data,
            steps_per_epoch,
            epochs
    ):
        """ train """
        self.history = self.model.fit(
            train_dataset,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_data,
            epochs=epochs,
        )

    def generate(self, text: str, p: float = 0.2):
        """generate"""
        input_tokens = self.tokenizer([text])
        packed_tokens = self.start_packer(input_tokens)
        token_length = tf.where(packed_tokens != 0)[-1, 1]
        initial_sequence_length = token_length + 1
        gen_ittr = self._generate_step(
            tokens=packed_tokens,
            p=p,
            start_index=initial_sequence_length  # 次に予測する位置
        )
        generated_text_parts = []
        for word in gen_ittr:
            generated_text_parts.append(word)
            print(word, end=" ")

        return "".join(generated_text_parts)  # より自然な表示のため

    def _generate_step(self, tokens, p=0.2, start_index=1):
        tokens = tokens.numpy()
        for i in range(start_index, self.maxlen):
            logits = self.model.predict([tokens], verbose=0)[:, i - 1, :]
            logits = tf.constant(logits)
            sampled_token = top_p_sample(logits[0], p).numpy()
            tokens[0][i] = sampled_token
            next_word = self.tokenizer.detokenize([sampled_token]).numpy().decode("utf-8")
            print(next_word)
            yield next_word
            if sampled_token == 2:  # EOS token
                raise StopIteration


def _build_token_dataset():
    """
    for create dataset to train tokenizer
    if you want to train tokenizer local,

    ds = _build_token_dataset()
    Run Transformer.train_tokenizer(ds)
    """
    # Data
    BATCH_SIZE = 64
    MIN_TRAINING_SEQ_LEN = 512

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


def top_p_sample(logits, p=0.2):
    """top sample"""
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

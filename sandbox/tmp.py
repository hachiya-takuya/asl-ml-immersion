"""here it is"""
import numpy as np
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
    GlobalAveragePooling1D
)


import utils_preproc


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
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        """call """
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


class Transformer(keras.Model):
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
        self.maxlen = maxlen
        inputs = Input(shape=(maxlen,))
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        x = self.embedding_layer(inputs)
        for _ in range(loop_n):
            transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
            x = transformer_block(x)

        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.1)(x)
        x = Dense(ff_dim, activation="relu")(x)
        x = Dropout(0.1)(x)
        outputs = Dense(vocab_size, activation="softmax")(x)

        super().__init__(inputs=inputs, outputs=outputs)
        self.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.tokenizer = tokenizer

    def create_tokenizer(self, data):
        """create tokenizer"""
        _, self.tokenizer = utils_preproc.tokenize(data)

    def train(
            self,
            *,
            x_train,
            y_train,
            batch_size=32,
            epochs=2,
            validation_data: tuple,
    ):
        """ train """
        self.history = self.fit(
            x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=validation_data
        )

    def generate(self, text: str):
        """generate"""
        text = "<start> " + text
        tokens = utils_preproc.tokenize([text])[0]
        gen_ittr = self._generate_step(tokens)
        for word in gen_ittr:
            print(word, end=" ")

    def _generate_step(self, tokens: list):
        for _ in range(self.maxlen):
            next_token_probabilities = self.predict([tokens])
            sampled_token = np.argmax(next_token_probabilities[:, -1], axis=-1)[0]
            tokens.append(sampled_token)
            next_word = utils_preproc.int2word(self.tokenizer, sampled_token)
            yield next_word
            if sampled_token == 2:
                raise StopIteration

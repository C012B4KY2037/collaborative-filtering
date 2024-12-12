import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable()
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_books, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.num_users = num_users
        self.num_books = num_books
        self.embedding_size = embedding_size

        # Embedding layers
        self.user_embedding = layers.Embedding(
            input_dim=num_users,
            output_dim=embedding_size,
            embeddings_initializer='glorot_uniform',
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
        )
        self.user_bias = layers.Embedding(input_dim=num_users, output_dim=1)

        self.book_embedding = layers.Embedding(
            input_dim=num_books,
            output_dim=embedding_size,
            embeddings_initializer='glorot_uniform',
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
        )
        self.book_bias = layers.Embedding(input_dim=num_books, output_dim=1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        book_vector = self.book_embedding(inputs[:, 1])
        book_bias = self.book_bias(inputs[:, 1])
        dot_user_book = tf.tensordot(user_vector, book_vector, axes=2)
        x = dot_user_book + user_bias + book_bias
        return tf.nn.sigmoid(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_users': self.num_users,
            'num_books': self.num_books,
            'embedding_size': self.embedding_size,
        })
        return config

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

def get_spatial_graph(num_node=25):
    """Constructs the physical adjacency matrix for 25 body joints."""
    A = np.zeros((num_node, num_node))
    edges = [(0, 1), (1, 20), (20, 2), (2, 3),
             (20, 4), (4, 5), (5, 6), (6, 7), (6, 21), (7, 22),
             (20, 8), (8, 9), (9, 10), (10, 11), (10, 23), (11, 24),
             (0, 12), (12, 13), (13, 14), (14, 15),
             (0, 16), (16, 17), (17, 18), (18, 19)]
             
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
        
    for i in range(num_node):
        A[i, i] = 1
        
    rowsum = A.sum(1)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    D = np.diag(r_inv)
    A_normalized = D.dot(A)
    
    # Return as a raw Numpy array instead of a tf.constant
    return A_normalized.astype(np.float32)

class GraphConvolution(layers.Layer):
    """Spatial graph convolution layer utilizing TensorFlow operations."""
    def __init__(self, out_channels, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.out_channels = out_channels
        
    def build(self, input_shape):
        self.conv = layers.Conv2D(self.out_channels, kernel_size=(1, 1), padding='same')
        
        # Register the matrix as a tracked, non-trainable weight in the Keras layer
        A_numpy = get_spatial_graph(25)
        self.A = self.add_weight(
            name="adjacency_matrix",
            shape=A_numpy.shape,
            initializer=tf.keras.initializers.Constant(A_numpy),
            trainable=False
        )
        super(GraphConvolution, self).build(input_shape)

    def call(self, x):
        # x shape: (Batch, Time, Vertices, Channels)
        x = self.conv(x)
        # Graph aggregation using adjacency matrix via einsum
        x = tf.einsum('ntvc,vw->ntwc', x, self.A)
        return x

class STGCN(Model):
    """Native TensorFlow Spatial-Temporal Graph Convolutional Network."""
    def __init__(self, num_classes=2, **kwargs):
        super(STGCN, self).__init__(**kwargs)
        
        self.spatial_gcn1 = GraphConvolution(32)
        self.temporal_conv1 = layers.Conv2D(64, kernel_size=(9, 1), padding='same')
        
        self.spatial_gcn2 = GraphConvolution(64)
        self.temporal_conv2 = layers.Conv2D(128, kernel_size=(9, 1), padding='same')
        
        self.pool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        x = tf.nn.relu(self.spatial_gcn1(x))
        x = tf.nn.relu(self.temporal_conv1(x))
        
        x = tf.nn.relu(self.spatial_gcn2(x))
        x = tf.nn.relu(self.temporal_conv2(x))
        
        x = self.pool(x)
        out = self.fc(x)
        return out
import keras
import matplotlib.pyplot as plt
from keras import layers, ops
import numpy as np
import tensorflow as tf
import math
from einops import rearrange

class RMSNorm(layers.Layer):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.w = self.add_weight((dim, ), "ones")
        self.eps = eps
    
    def _norm(self, x):
        return x * ops.rsqrt(ops.mean(ops.power(x, 2), -1, keepdims=True) + self.eps)
    
    def call(self, x):
        output = self._norm(ops.cast(x, "float32"))
        output = ops.cast(output, x.dtype)
        return output * self.w

def selective_scan(u, delta, A, B, C, D):
    # first step of A_bar = exp(ΔA), i.e., ΔA
    dA = ops.einsum('bld,dn->bldn', delta, A) 
    dB_u = ops.einsum('bld,bld,bln->bldn', delta, u, B)
    
    dA_cumsum = ops.pad(
        dA[:, 1:], [[0, 0], [1, 1], [0, 0], [0, 0]])[:, 1:, :, :]
    
    dA_cumsum = tf.reverse(dA_cumsum, axis=[1])  # Flip along axis 1
    
    # Cumulative sum along all the input tokens, parallel prefix sum, 
    # calculates dA for all the input tokens parallely
    dA_cumsum = ops.cumsum(dA_cumsum, axis=1)  

    # second step of A_bar = exp(ΔA), i.e., exp(ΔA)
    dA_cumsum = ops.exp(dA_cumsum)  
    dA_cumsum = tf.reverse(dA_cumsum, axis=[1])  # Flip back along axis 1

    x = dB_u * dA_cumsum
    # 1e-12 to avoid division by 0
    x = ops.cumsum(x, axis=1)/(dA_cumsum + 1e-12) 

    y = ops.einsum('bldn,bln->bld', x, C)
    
    return y + u * D 



class MambaBlock(layers.Layer):
    def __init__(self, internal_dim = 256, conv1d_use_bias = False, conv1d_kernel_size = 3, model_states = 32, modelInputDims = 128, outDenseBias = False):
        super().__init__()
        

        self.internal_dim = internal_dim
        self.conv1d_use_bias = conv1d_use_bias
        self.conv1d_kernel_size = conv1d_kernel_size
        self.model_states = model_states
        self.modelInputDims = modelInputDims
        
        
        self.delta_t_rank = math.ceil(modelInputDims/16)

        self.in_poj = layers.Dense(
            internal_dim*2, use_bias=False
        )

        self.delta_poj = layers.Dense(
            internal_dim, use_bias=True
        )

        self.conv1d = layers.Conv1D(
            internal_dim,
            use_bias= conv1d_use_bias,
            kernel_size= conv1d_kernel_size,
            groups=internal_dim,
            data_format='channels_first',
            padding="causal"            
        )

        self.x_poj = layers.Dense(
            self.delta_t_rank+model_states*2, use_bias=False
        )

        initializerArrange = lambda shape, dtype : ops.repeat(
            ops.expand_dims(ops.arange(1, shape[1]+1, dtype= dtype), 0),
            shape[0],
            axis=0
        )
        
        self.A_log = self.add_variable(
            (internal_dim, model_states, ), initializer= initializerArrange, trainable=True
        )

        self.D = self.add_variable((internal_dim, ), "ones")
        
        self.out_poj = layers.Dense(
            modelInputDims,
            use_bias=outDenseBias
        )

        # print(self.A.shape)

    def call(self, x):
        (batch_size, seq_len, dimension) = x.shape

        x_and_res = self.in_poj(x)
        # print(x_and_res)

        # print(

        #     tf.split(
        #     x_and_res,
        #     [self.internal_dim,
        #      self.internal_dim],
        #     axis= -1
        #     )

        # )


        (x, res) = tf.split(
            x_and_res,
            [self.internal_dim,
             self.internal_dim],
            axis= -1
        )
        # print(p)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :seq_len]
        x = rearrange(x, 'b d_in l -> b l d_in')

        x = ops.nn.swish(x)
        y = self.ssm(x)
        y = y * ops.nn.swish(res)
        return self.out_poj(y)


    def ssm(self, x):
        (d_in, n) = self.A_log.shape

        A = -ops.exp(ops.cast(self.A_log, "float32"))
        D = ops.cast(self.D, "float32")

        x_dbl = self.x_poj(x)

        (delta, B, C) = tf.split(
            x_dbl,
            [self.delta_t_rank, n, n],
            axis= -1
        )

        delta = ops.nn.softplus(self.delta_poj(delta))

        return selective_scan(x, delta, A, B, C, D)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

class GPT_Embedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = positional_encoding(maxlen, embed_dim)

    def call(self, x):
        maxlen = ops.shape(x)[-1]
        positions = self.pos_emb[:, :maxlen]
        x = self.token_emb(x)
        return x + positions

x = np.random.randn(1, 512, 128)
# print(RMSNorm(10)(x))

# print(MambaBlock()(x))

vocab_size = 20000  # Only consider the top 20k words
maxlen = 128  # Max sequence size
embed_dim = 256  # Embedding size for each token
num_heads = 2  # Number of attention heads
feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer
N_layers = 24

def init_model():
    
    input_layer = layers.Input(shape=(None,), name='input_ids')
    # embedding_layer = GPT_Embedding(maxlen, vocab_size, embed_dim)
    embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
    x = embedding_layer(input_layer)

    for i in range(N_layers):
        x = MambaBlock(modelInputDims=embed_dim, internal_dim=128, model_states=64, )(x)
        x = layers.Dropout(0.4)(x)
        x = RMSNorm(embed_dim)(x)

    outputL = layers.Dense(vocab_size)(x)

    return keras.Model([input_layer], [outputL])

model = init_model()
model.summary()

model.compile("adam", loss= keras.losses.SparseCategoricalCrossentropy(True))
model.fit(np.random.randint(0, vocab_size, (3, 512)), np.random.randint(0, vocab_size, (3, 512)))
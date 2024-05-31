import tensorflow as tf
from transformers import TFSwinForImageClassification
from tf_keras import layers, Model

class PatchMerging(layers.Layer):
    def __init__(self, input_resolution, dim, **kwargs):
        super().__init__(**kwargs)
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = layers.Dense(2 * dim, use_bias=False)
        self.norm = layers.LayerNormalization(epsilon=1e-5)
    
    def call(self, x):
        H, W = self.input_resolution
        x = tf.reshape(x, (-1, H, W, self.dim))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat([x0, x1, x2, x3], axis=-1)
        x = tf.reshape(x, (-1, (H // 2) * (W // 2), 4 * self.dim))
        x = self.norm(x)
        x = self.reduction(x)
        return x

class WindowAttention(layers.Layer):
    def __init__(self, dim, window_size, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = layers.Dense(dim * 3, use_bias=False)
        self.attn = layers.Dense(dim)

    def call(self, x, mask=None):
        B, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        qkv = tf.reshape(self.qkv(x), (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2]))
        if mask is not None:
            nW = tf.shape(mask)[0]
            attn = tf.reshape(attn, (B // nW, nW, self.num_heads, N, N)) + mask
            attn = tf.reshape(attn, (-1, self.num_heads, N, N))

        attn = tf.nn.softmax(attn, axis=-1)
        attn = tf.nn.dropout(attn, rate=0.0)
        x = tf.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (B, N, C))

        x = self.attn(x)
        return x

class SwinTransformerBlock(layers.Layer):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4., **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)

        self.drop_path = tf.identity
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.mlp = tf.keras.Sequential([
            layers.Dense(int(dim * mlp_ratio), activation='gelu'),
            layers.Dense(dim)
        ])

    def build(self, input_shape):
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.attn_mask = self.create_mask(self.input_resolution)

    def create_mask(self, x_shape):
        if self.shift_size > 0:
            H, W = x_shape
            img_mask = tf.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = self.window_partition(img_mask)
            mask_windows = tf.reshape(mask_windows, (-1, self.window_size * self.window_size))
            attn_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
        else:
            attn_mask = None

        return attn_mask

    def window_partition(self, x):
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        C = tf.shape(x)[3]
        x = tf.reshape(x, (B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C))
        windows = tf.reshape(tf.transpose(x, perm=[0, 1, 3, 2, 4, 5]), (-1, self.window_size, self.window_size, C))
        return windows

    def window_reverse(self, windows, H, W):
        B = tf.shape(windows)[0] // (H // self.window_size * W // self.window_size)
        x = tf.reshape(windows, (B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, -1))
        x = tf.transpose(tf.reshape(x, (B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, -1)), perm=[0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, (B, H, W, -1))
        return x

    def call(self, x):
        H, W = self.input_resolution
        B = tf.shape(x)[0]
        L = tf.shape(x)[1]
        C = tf.shape(x)[2]
        
        # Perform the check using TensorFlow operations
        expected_L = H * W
        tf.debugging.assert_equal(L, expected_L, message=f"Input feature has wrong size. Expected {H * W}, got {L}")

        shortcut = x
        x = tf.reshape(x, (B, H, W, C))

        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        else:
            shifted_x = x

        x_windows = self.window_partition(shifted_x)
        x_windows = tf.reshape(x_windows, (-1, self.window_size * self.window_size, C))
        attn_windows = self.attn(self.norm1(x_windows), mask=self.attn_mask)
        attn_windows = tf.reshape(attn_windows, (-1, self.window_size, self.window_size, C))

        shifted_x = self.window_reverse(attn_windows, H, W)
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2])
        else:
            x = shifted_x

        x = tf.reshape(x, (B, H * W, C))
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
    
class PatchExtract(layers.Layer):
    def __init__(self, patch_size=(4, 4), **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size[0], self.patch_size[1], 1],
            strides=[1, self.patch_size[0], self.patch_size[1], 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1]
        return tf.reshape(patches, (batch_size, patch_num * patch_num, patch_dim))
    
def build_swin_transformer(input_shape=(224, 224, 3), num_classes=1000):
    input_layer = layers.Input(shape=input_shape)
    x = PatchExtract(patch_size=(4, 4))(input_layer)  # Apply patch extraction inside the model
    x = layers.Dense(96)(x)  # Linear embedding
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    
    # Stage 1
    for _ in range(2):
        x = SwinTransformerBlock(96, input_resolution=(56, 56), num_heads=3)(x)
    
    # Stage 2
    x = PatchMerging((56, 56), 96)(x)
    for _ in range(2):
        x = SwinTransformerBlock(192, input_resolution=(28, 28), num_heads=6)(x)
    
    # Stage 3
    x = PatchMerging((28, 28), 192)(x)
    for _ in range(6):
        x = SwinTransformerBlock(384, input_resolution=(14, 14), num_heads=12)(x)
    
    # Stage 4
    x = PatchMerging((14, 14), 384)(x)
    for _ in range(2):
        x = SwinTransformerBlock(768, input_resolution=(7, 7), num_heads=24)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    output_layer = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

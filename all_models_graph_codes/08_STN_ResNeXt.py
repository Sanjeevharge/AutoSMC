# ================================================================
#  STN-ResNeXt — RADIOML 2016.10A
#  Perenda et al. 2021 [48]:
#    "STN introduced INTO ResNeXt to enhance robustness to
#     signal shape transformations"
#  Architecture:
#    1D-STN (localisation → grid → sampler) on input (128,2)
#    → Conv1D(64,3) + BN + ReLU
#    → 3 × ResNeXt grouped-conv blocks [64,128,128] + MaxPool1D(2)
#    → GlobalAveragePooling1D → Dense(nc,softmax)
#  Norm: per-sample max  |  lr=0.001  |  per-SNR training
# ================================================================
import pickle, numpy as np, random, tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATASET_PATH = "/kaggle/input/datasets/sanjeevharge/2016-10a/RML2016.10a_dict.pkl"
SNRS = list(range(-20, 8, 2))

def set_seed(s=42):
    np.random.seed(s); tf.random.set_seed(s); random.seed(s)

def load_raw(path):
    with open(path, 'rb') as f: data = pickle.load(f, encoding='latin1')
    mods = sorted(set(k[0] for k in data.keys()))
    cmap = {m: i for i, m in enumerate(mods)}; nc = len(mods)
    print(f"Classes ({nc}): {mods}")
    dbs = {}
    for snr in SNRS:
        Xa, ya = [], []
        for mod in mods:
            X = np.transpose(data[(mod, snr)], (0, 2, 1)).astype(np.float32)
            Xa.append(X); ya.extend([cmap[mod]] * len(X))
        Xa = np.vstack(Xa); ya = np.array(ya)
        Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=0.2,
                                               stratify=ya, random_state=42)
        dbs[snr] = (Xtr, Xte, ytr, yte)
    return dbs, nc

def norm_per_sample(Xtr, Xte):
    def _n(X):
        m = np.max(np.abs(X), axis=(1, 2), keepdims=True)
        return X / np.where(m == 0, 1.0, m)
    return _n(Xtr), _n(Xte)

def train_one(model, Xtr, ytr, Xte, yte, lr, epochs, bs):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cb = [
        tf.keras.callbacks.ReduceLROnPlateau('val_loss', 0.5, 5,
                                              min_lr=1e-9, verbose=0),
        tf.keras.callbacks.EarlyStopping('val_accuracy', patience=15,
                                          restore_best_weights=True, verbose=0),
    ]
    model.fit(Xtr, ytr, epochs=epochs, batch_size=bs,
              validation_data=(Xte, yte), callbacks=cb, verbose=0)
    return float(accuracy_score(yte,
                  np.argmax(model.predict(Xte, verbose=0), axis=1)))

# ── 1D Spatial Transformer Network (STN) ──────────────────────
# Localisation net predicts [scale, shift] for 1D signal warping
class STN1D(tf.keras.layers.Layer):
    """1-D Spatial Transformer: learns a scale+shift warp of the signal."""
    def __init__(self, seq_len=128, **kw):
        super().__init__(**kw)
        self.seq_len = seq_len
        # localisation network (small CNN → 2 params: scale, shift)
        self.loc = tf.keras.Sequential([
            layers.Conv1D(32, 3, padding='same', activation='relu'),
            layers.MaxPool1D(2),
            layers.Conv1D(32, 3, padding='same', activation='relu'),
            layers.MaxPool1D(2),
            layers.Flatten(),
            layers.Dense(32, activation='relu'),
            layers.Dense(2, bias_initializer='zeros',
                         kernel_initializer='zeros')   # predicts [scale, shift]
        ])

    def call(self, x):
        # x: (B, T, C)
        B = tf.shape(x)[0]
        params = self.loc(x)          # (B, 2)
        scale  = params[:, 0:1] + 1.0 # centre around 1 (identity)
        shift  = params[:, 1:2]        # centre around 0

        # source grid: uniform positions in [-1, 1]
        src = tf.linspace(-1.0, 1.0, self.seq_len)  # (T,)
        src = tf.tile(src[tf.newaxis, :], [B, 1])   # (B, T)
        # transformed positions
        tgt = scale * src + shift                    # (B, T)
        # clamp to [-1, 1]
        tgt = tf.clip_by_value(tgt, -1.0, 1.0)
        # bilinear (linear) sampling
        tgt_idx = (tgt + 1.0) / 2.0 * tf.cast(self.seq_len - 1, tf.float32)
        idx0 = tf.cast(tf.floor(tgt_idx), tf.int32)
        idx1 = tf.minimum(idx0 + 1, self.seq_len - 1)
        w1   = tgt_idx - tf.cast(idx0, tf.float32)   # (B, T)
        w0   = 1.0 - w1

        # gather from all channels
        C = x.shape[-1] if x.shape[-1] is not None else tf.shape(x)[-1]
        # x: (B, T, C) → gather
        def gather_ch(args):
            xi, i0, i1, a0, a1 = args
            # xi: (T,C), i0/i1: (T,), a0/a1: (T,)
            v0 = tf.gather(xi, i0)   # (T, C)
            v1 = tf.gather(xi, i1)   # (T, C)
            return a0[:, tf.newaxis] * v0 + a1[:, tf.newaxis] * v1  # (T,C)

        out = tf.map_fn(gather_ch, (x, idx0, idx1, w0, w1),
                        fn_output_signature=tf.float32)  # (B, T, C)
        return out

# ── ResNeXt block (grouped convolutions via DepthwiseConv1D) ───
def resnext_block(x, out_filters, cardinality=4):
    group_width = out_filters // cardinality
    shortcut = layers.Conv1D(out_filters, 1, padding='same')(x)
    paths = []
    for _ in range(cardinality):
        p = layers.Conv1D(group_width, 1, padding='same')(x)
        p = layers.DepthwiseConv1D(3, padding='same')(p)
        p = layers.Conv1D(group_width, 1, padding='same')(p)
        paths.append(p)
    agg = layers.Concatenate()(paths)            # (T, out_filters)
    agg = layers.BatchNormalization()(agg)
    out = layers.Add()([agg, shortcut])
    return layers.ReLU()(out)

def build_model(nc, seq_len=128):
    inp = tf.keras.Input(shape=(seq_len, 2))
    # STN
    x   = STN1D(seq_len=seq_len)(inp)
    # Stem conv
    x   = layers.Conv1D(64, 3, padding='same')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.ReLU()(x)
    # 3 ResNeXt blocks
    for f in [64, 128, 128]:
        x = resnext_block(x, f, cardinality=4)
        x = layers.MaxPool1D(2)(x)
    x   = layers.GlobalAveragePooling1D()(x)
    return tf.keras.Model(inp, layers.Dense(nc, activation='softmax')(x))

set_seed(42)
dbs, nc = load_raw(DATASET_PATH)
accs = []
print("Training STN-ResNeXt — one fresh model per SNR")
for snr in SNRS:
    Xtr_r, Xte_r, ytr, yte = dbs[snr]
    Xtr_n, Xte_n = norm_per_sample(Xtr_r, Xte_r)
    set_seed(42); tf.keras.backend.clear_session()
    acc = train_one(build_model(nc), Xtr_n, ytr, Xte_n, yte,
                    lr=1e-3, epochs=100, bs=128)
    accs.append(acc)
    print(f"  SNR {snr:+4d} dB  →  {acc*100:.2f}%")

print("\nFull results:")
for s, a in zip(SNRS, accs):
    print(f"  SNR {s:+4d} dB  →  {a*100:.2f}%")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(SNRS, [a*100 for a in accs],
        marker='h', color='#e6c619', linewidth=1.8,
        markersize=6, label='STN-ResNeXt')
ax.set_xlabel("SNR(dB)", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("STN-ResNeXt — RADIOML 2016.10A", fontsize=11)
ax.set_xticks(SNRS)
ax.tick_params(axis='x', labelsize=8)
ax.set_ylim(0, 100)
ax.set_yticks(range(0, 101, 10))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}%"))
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("STN_ResNeXt_accuracy.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved → STN_ResNeXt_accuracy.png")

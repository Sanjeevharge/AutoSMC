# ================================================================
#  AutoSMC — RADIOML 2016.10A
#  Wang et al., IEEE TIFS 2024  —  Table V optimal architecture
#
#  IMPORTANT NOTE on learning rate:
#  Table V shows lr=1e-5. That is the LR at the FINAL NAS trial,
#  after dozens of trials where the model inherited weights from
#  previous trials (network morphism). For cold-start training
#  from random init we use lr=1e-3 (Adam default, same as all
#  baselines) and let ReduceLROnPlateau decay it naturally.
#  This matches the paper's intent of "Adam, lr set by default"
#  for the comparison experiments.
#
#  Norm: global-max (same as MsmcNet, per paper Sec IV-A-2)
#  Per-SNR training: separate fresh model for each SNR point
# ================================================================
import pickle, numpy as np, random, tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATASET_PATH = "/kaggle/input/datasets/sanjeevharge/2016-10a/RML2016.10a_dict.pkl"
SNRS = list(range(-20, 8, 2))   # -20, -18, ..., +6  (14 points)

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

def norm_global(Xtr, Xte):
    g = np.max(np.abs(Xtr))
    return Xtr / g, Xte / g

def train_one(model, Xtr, ytr, Xte, yte, lr, epochs, bs):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    cb = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=8,
            min_lr=1e-9, verbose=0),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=25,
            restore_best_weights=True, verbose=0),
    ]
    model.fit(Xtr, ytr, epochs=epochs, batch_size=bs,
              validation_data=(Xte, yte), callbacks=cb, verbose=0)
    return float(accuracy_score(yte,
                  np.argmax(model.predict(Xte, verbose=0), axis=1)))

# ── RFF Layer (non-trainable random Fourier features) ──────────
class RFFLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, scale, **kw):
        super().__init__(**kw)
        self.output_dim = output_dim
        self.scale      = scale
    def build(self, s):
        d = s[-1]
        self.W = self.add_weight(
            (d, self.output_dim),
            initializer=tf.random_normal_initializer(stddev=self.scale),
            trainable=False, name='W')
        self.b = self.add_weight(
            (self.output_dim,),
            initializer=tf.random_uniform_initializer(0, 2*np.pi),
            trainable=False, name='b')
        super().build(s)
    def call(self, x):
        return (tf.sqrt(2.0 / float(self.output_dim)) *
                tf.cos(tf.matmul(x, self.W) + self.b))

# ── Table V: optimal structure searched at SNR=6dB ─────────────
# (sigma, filter_list, kernel_l, delta, rff_dims, rff_scales, w)
CRFF_CFG = [
    (3, [128, 128,  64], 3, 5, [2048, 2048, 1024, 512, 4096], [10, 15, 10, 15, 10], 0.001),
    (3, [128,  64, 128], 3, 1, [8192],                         [15],                 0.0),
    (2, [ 32,  32],      3, 3, [2048, 512, 2048],              [15, 15, 13],          0.1),
    (3, [ 64, 128,  32], 7, 1, [2048],                         [10],                 0.0),
]

def build_model(nc):
    inp = tf.keras.Input(shape=(128, 2))

    # ── Conv Head (IQF-style, Table V: sH=128, lH=7) ───────────
    # Reshape to (128,2,1) then Conv2D with kernel (7,2) — the
    # width-2 kernel covers BOTH I and Q at once, fusing them.
    # valid padding: 128 - 7 + 1 = 122 time steps remain
    x = layers.Reshape((128, 2, 1))(inp)
    x = layers.Conv2D(128, (7, 2), padding='valid')(x)   # → (122,1,128)
    x = layers.Reshape((122, 128))(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPool1D(2)(x)                           # → (61,128)

    # ── 4 CRFF Blocks ──────────────────────────────────────────
    for _, flist, lk, _, rdims, scales, w in CRFF_CFG:
        out_f = flist[-1]

        # Conv branch: sigma sub-blocks of Conv→BN→LeakyReLU
        conv = x
        for f in flist:
            conv = layers.Conv1D(f, lk, padding='same')(conv)
            conv = layers.BatchNormalization()(conv)
            conv = layers.LeakyReLU()(conv)

        # Align skip dim for RFF branch
        if x.shape[-1] != out_f:
            x = layers.Conv1D(out_f, 1, padding='same')(x)

        # RFF branch (only when w > 0)
        if w > 0:
            rff = x
            for od, sc in zip(rdims, scales):
                rff = RFFLayer(od, sc)(rff)
            rff = layers.Dense(out_f)(rff)
            x   = conv + w * rff
        else:
            x = conv

        x = layers.MaxPool1D(2, padding='same')(x)

    x = layers.GlobalAveragePooling1D()(x)
    return tf.keras.Model(inp, layers.Dense(nc, activation='softmax')(x))

# ── Per-SNR training ────────────────────────────────────────────
set_seed(42)
dbs, nc = load_raw(DATASET_PATH)
accs = []
print("Training AutoSMC — one fresh model per SNR")
print("lr=1e-3 (Adam default for cold-start), decayed by ReduceLROnPlateau")
for snr in SNRS:
    Xtr_r, Xte_r, ytr, yte = dbs[snr]
    Xtr_n, Xte_n = norm_global(Xtr_r, Xte_r)
    set_seed(42); tf.keras.backend.clear_session()
    acc = train_one(build_model(nc), Xtr_n, ytr, Xte_n, yte,
                    lr=1e-3,   # Adam default — correct for cold-start
                    epochs=200,
                    bs=128)
    accs.append(acc)
    print(f"  SNR {snr:+4d} dB  →  {acc*100:.2f}%")

print("\nFull results:")
for s, a in zip(SNRS, accs):
    print(f"  SNR {s:+4d} dB  →  {a*100:.2f}%")

# ── Plot matching paper Fig 2 axes exactly ──────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(SNRS, [a*100 for a in accs],
        marker='s', color='#c00040', linewidth=1.8,
        markersize=5, label='AutoSMC')
ax.set_xlabel("SNR(dB)", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("AutoSMC — RADIOML 2016.10A", fontsize=11)
ax.set_xticks(SNRS)
ax.tick_params(axis='x', labelsize=8)
ax.set_ylim(0, 100)
ax.set_yticks(range(0, 101, 10))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}%"))
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("AutoSMC_accuracy.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved → AutoSMC_accuracy.png")

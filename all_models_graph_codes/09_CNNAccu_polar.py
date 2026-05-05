# ================================================================
#  CNNAccu_polar — RADIOML 2016.10A
#  Teng et al. 2020 [12]:
#    "SMC method based on ACCUMULATED POLAR FEATURES
#     with a channel compensation mechanism"
#  Paper says input is raw IQ (128,2), polar conversion is INSIDE model:
#    Step 1: IQ (128,2) → amplitude = sqrt(I²+Q²), phase = arctan2(Q,I)
#    Step 2: Accumulate across time (cumsum) → (128,2) polar accumulated
#    Step 3: Concatenate original IQ + accumulated polar → (128,4)
#    Step 4: CNN classification head
#  Architecture:
#    PolarAccumLayer (custom, non-trainable)
#    → Conv1D(64,3,same)+BN+ReLU × 2 → MaxPool1D(2)
#    → Conv1D(128,3,same)+BN+ReLU × 2 → GAP
#    → Dense(128,relu) → Dropout(0.5) → Dense(nc,softmax)
#  Norm: per-sample max on IQ, applied BEFORE polar conversion
#  lr=0.001  |  per-SNR training
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

# ── Accumulated Polar Feature Layer (non-trainable) ─────────────
class PolarAccumLayer(tf.keras.layers.Layer):
    """
    Converts raw IQ to accumulated polar features:
      I,Q → amplitude = sqrt(I²+Q²),  phase = arctan2(Q,I)
      accumulated amplitude = cumsum(amplitude) along time
      accumulated phase     = cumsum(phase)     along time
    Output: concatenate [I, Q, cumsum_amp, cumsum_phase] → (T, 4)
    """
    def call(self, x):
        I   = x[:, :, 0:1]                          # (B,128,1)
        Q   = x[:, :, 1:2]                          # (B,128,1)
        amp = tf.sqrt(I**2 + Q**2)                  # (B,128,1)
        phi = tf.atan2(Q, I)                         # (B,128,1)
        cum_amp = tf.cumsum(amp, axis=1)             # (B,128,1)
        cum_phi = tf.cumsum(phi, axis=1)             # (B,128,1)
        return tf.concat([I, Q, cum_amp, cum_phi], axis=-1)  # (B,128,4)

def build_model(nc):
    inp = tf.keras.Input(shape=(128, 2))
    # Polar accumulated feature extraction
    x   = PolarAccumLayer()(inp)                   # (B,128,4)
    # CNN block 1
    x   = layers.Conv1D(64, 3, padding='same')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.ReLU()(x)
    x   = layers.Conv1D(64, 3, padding='same')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.ReLU()(x)
    x   = layers.MaxPool1D(2)(x)
    # CNN block 2
    x   = layers.Conv1D(128, 3, padding='same')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.ReLU()(x)
    x   = layers.Conv1D(128, 3, padding='same')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.ReLU()(x)
    x   = layers.GlobalAveragePooling1D()(x)
    # FC head
    x   = layers.Dense(128, activation='relu')(x)
    x   = layers.Dropout(0.5)(x)
    return tf.keras.Model(inp, layers.Dense(nc, activation='softmax')(x))

set_seed(42)
dbs, nc = load_raw(DATASET_PATH)
accs = []
print("Training CNNAccu_polar — one fresh model per SNR")
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
        marker='o', color='#e07820', linewidth=1.8,
        markersize=5, label='CNN_Accu_polar')
ax.set_xlabel("SNR(dB)", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("CNN_Accu_polar — RADIOML 2016.10A", fontsize=11)
ax.set_xticks(SNRS)
ax.tick_params(axis='x', labelsize=8)
ax.set_ylim(0, 100)
ax.set_yticks(range(0, 101, 10))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}%"))
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("CNNAccu_polar_accuracy.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved → CNNAccu_polar_accuracy.png")

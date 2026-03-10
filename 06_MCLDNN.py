# ================================================================
#  MCLDNN — RADIOML 2016.10A
#  Xu et al. 2020 [47]: Spatiotemporal multi-channel learning
#  Input: raw IQ (128,2), I-component (128,1), Q-component (128,1)
#  Architecture:
#    1D branch: Conv1D(50,8,same,relu) × 2  → (128,50)
#    2D branch: Reshape(128,2,1) → Conv2D(50,(8,1),same,relu) → Reshape(128,100)
#    Concat → (128,150)
#    LSTM(128,seq) → LSTM(128) → Dense(256,relu) → Dropout(0.5) → Dense(nc,softmax)
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

def build_model(nc):
    inp = tf.keras.Input(shape=(128, 2))
    # 1D CNN branch on raw IQ
    x1  = layers.Conv1D(50, 8, padding='same', activation='relu')(inp)
    x1  = layers.Conv1D(50, 8, padding='same', activation='relu')(x1)  # (128,50)
    # 2D CNN branch: treats (128,2) as 2D image
    x2  = layers.Reshape((128, 2, 1))(inp)
    x2  = layers.Conv2D(50, (8, 1), padding='same', activation='relu')(x2)
    x2  = layers.Reshape((128, 100))(x2)                               # (128,100)
    # Concatenate → (128,150)
    x   = layers.Concatenate(axis=-1)([x1, x2])
    x   = layers.LSTM(128, return_sequences=True)(x)
    x   = layers.LSTM(128)(x)
    x   = layers.Dense(256, activation='relu')(x)
    x   = layers.Dropout(0.5)(x)
    return tf.keras.Model(inp, layers.Dense(nc, activation='softmax')(x))

set_seed(42)
dbs, nc = load_raw(DATASET_PATH)
accs = []
print("Training MCLDNN — one fresh model per SNR")
for snr in SNRS:
    Xtr_r, Xte_r, ytr, yte = dbs[snr]
    Xtr_n, Xte_n = norm_per_sample(Xtr_r, Xte_r)
    set_seed(42); tf.keras.backend.clear_session()
    acc = train_one(build_model(nc), Xtr_n, ytr, Xte_n, yte,
                    lr=1e-3, epochs=100, bs=64)
    accs.append(acc)
    print(f"  SNR {snr:+4d} dB  →  {acc*100:.2f}%")

print("\nFull results:")
for s, a in zip(SNRS, accs):
    print(f"  SNR {s:+4d} dB  →  {a*100:.2f}%")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(SNRS, [a*100 for a in accs],
        marker='^', color='#ff7f0e', linewidth=1.8,
        markersize=5, label='MCLDNN')
ax.set_xlabel("SNR(dB)", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("MCLDNN — RADIOML 2016.10A", fontsize=11)
ax.set_xticks(SNRS)
ax.tick_params(axis='x', labelsize=8)
ax.set_ylim(0, 100)
ax.set_yticks(range(0, 101, 10))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}%"))
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("MCLDNN_accuracy.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved → MCLDNN_accuracy.png")

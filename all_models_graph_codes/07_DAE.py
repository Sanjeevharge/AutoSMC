# ================================================================
#  DAE — RADIOML 2016.10A
#  Ke & Vikalo 2022 [14]: LSTM Auto-Encoder for modulation classification
#  Paper: "trains the autoencoder based on LSTM layers AND the
#          classification network SIMULTANEOUSLY"
#  Architecture:
#    Encoder:    LSTM(128)                         → latent (128,)
#    Decoder:    RepeatVector(128) → LSTM(128,seq) → TimeDistributed(Dense(2))
#    Classifier: Dense(256,relu) → Dropout(0.5)   → Dense(nc,softmax)
#  Loss: joint = cross_entropy + λ * MSE(reconstruction)
#  Norm: per-sample max  |  lr=0.001  |  per-SNR training
# ================================================================
import pickle, numpy as np, random, tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATASET_PATH = "/kaggle/input/datasets/sanjeevharge/2016-10a/RML2016.10a_dict.pkl"
SNRS = list(range(-20, 8, 2))
LAMBDA_RECON = 0.5   # reconstruction loss weight

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

def build_model(nc):
    inp = tf.keras.Input(shape=(128, 2))
    # Encoder
    encoded      = layers.LSTM(128)(inp)                              # (128,)
    # Decoder branch (reconstruction)
    dec          = layers.RepeatVector(128)(encoded)                  # (128,128)
    dec          = layers.LSTM(128, return_sequences=True)(dec)       # (128,128)
    reconstructed= layers.TimeDistributed(layers.Dense(2),
                                          name='reconstruction')(dec) # (128,2)
    # Classifier branch
    clf          = layers.Dense(256, activation='relu')(encoded)
    clf          = layers.Dropout(0.5)(clf)
    clf_out      = layers.Dense(nc, activation='softmax',
                                name='classification')(clf)
    return Model(inp, [reconstructed, clf_out])

def train_one(model, Xtr, ytr, Xte, yte, lr, epochs, bs):
    # Joint loss: classification (CE) + reconstruction (MSE)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss={
            'reconstruction': 'mse',
            'classification': 'sparse_categorical_crossentropy'
        },
        loss_weights={
            'reconstruction': LAMBDA_RECON,
            'classification': 1.0
        },
        metrics={'classification': 'accuracy'}
    )
    cb = [
        tf.keras.callbacks.ReduceLROnPlateau('val_loss', 0.5, 5,
                                              min_lr=1e-9, verbose=0),
        tf.keras.callbacks.EarlyStopping('val_classification_accuracy',
                                          patience=15,
                                          restore_best_weights=True, verbose=0),
    ]
    # targets: reconstruction target = input itself, classification = labels
    model.fit(Xtr, {'reconstruction': Xtr, 'classification': ytr},
              epochs=epochs, batch_size=bs,
              validation_data=(Xte, {'reconstruction': Xte, 'classification': yte}),
              callbacks=cb, verbose=0)
    _, clf_out = model.predict(Xte, verbose=0)
    return float(accuracy_score(yte, np.argmax(clf_out, axis=1)))

set_seed(42)
dbs, nc = load_raw(DATASET_PATH)
accs = []
print("Training DAE — one fresh model per SNR (joint AE + classifier loss)")
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
        marker='o', color='#6baed6', linewidth=1.8,
        markersize=5, markerfacecolor='none', label='DAE')
ax.set_xlabel("SNR(dB)", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("DAE — RADIOML 2016.10A", fontsize=11)
ax.set_xticks(SNRS)
ax.tick_params(axis='x', labelsize=8)
ax.set_ylim(0, 100)
ax.set_yticks(range(0, 101, 10))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}%"))
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("DAE_accuracy.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved → DAE_accuracy.png")

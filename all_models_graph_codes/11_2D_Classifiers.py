# ================================================================
#  2D Classifiers — RADIOML 2016.10A
#  MobileNet [44], ResNet50 [43], DenseNet169 [45]
#  Paper: "input of MobileNet, ResNet50, DenseNet169 are 16 copies
#          of the raw IQ signal; input dimension is (128,32)"
#  Preprocessing: per-sample max norm → tile 16× along channel
#                 → reshape to (128,32,1)
#  Weights: None (random init, trained from scratch)
#  lr=0.001  |  batch_size=64 (MobileNet,ResNet50), 32 (DenseNet169)
#  per-SNR training
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

def norm_and_tile(Xtr, Xte):
    """Per-sample max norm, then tile 16× along feature dim → (N,128,32,1)."""
    def _n(X):
        m = np.max(np.abs(X), axis=(1, 2), keepdims=True)
        return X / np.where(m == 0, 1.0, m)
    def _tile(X):
        return np.tile(_n(X), (1, 1, 16))[..., np.newaxis]  # (N,128,32,1)
    return _tile(Xtr), _tile(Xte)

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

def build_mobilenet(nc):
    base = tf.keras.applications.MobileNet(
        input_shape=(128, 32, 1), alpha=0.5,
        include_top=False, weights=None)
    inp = tf.keras.Input(shape=(128, 32, 1))
    x   = base(inp)
    x   = layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inp, layers.Dense(nc, activation='softmax')(x))

def build_resnet50(nc):
    base = tf.keras.applications.ResNet50(
        input_shape=(128, 32, 1),
        include_top=False, weights=None)
    inp = tf.keras.Input(shape=(128, 32, 1))
    x   = base(inp)
    x   = layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inp, layers.Dense(nc, activation='softmax')(x))

def build_densenet169(nc):
    base = tf.keras.applications.DenseNet169(
        input_shape=(128, 32, 1),
        include_top=False, weights=None)
    inp = tf.keras.Input(shape=(128, 32, 1))
    x   = base(inp)
    x   = layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inp, layers.Dense(nc, activation='softmax')(x))

# ── Model registry ──────────────────────────────────────────────
MODELS = [
    ("MobileNet",   build_mobilenet,   64,  '#b5a400', 's'),
    ("ResNet50",    build_resnet50,    64,  '#ff7f0e', '+'),
    ("DenseNet169", build_densenet169, 32,  '#2ca02c', 'D'),
]

set_seed(42)
dbs, nc = load_raw(DATASET_PATH)
all_accs = {}

for name, builder, bs, color, marker in MODELS:
    accs = []
    print(f"\nTraining {name} — one fresh model per SNR")
    for snr in SNRS:
        Xtr_r, Xte_r, ytr, yte = dbs[snr]
        Xtr_n, Xte_n = norm_and_tile(Xtr_r, Xte_r)
        set_seed(42); tf.keras.backend.clear_session()
        acc = train_one(builder(nc), Xtr_n, ytr, Xte_n, yte,
                        lr=1e-3, epochs=100, bs=bs)
        accs.append(acc)
        print(f"  SNR {snr:+4d} dB  →  {acc*100:.2f}%")
    all_accs[name] = accs

    # Individual plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(SNRS, [a*100 for a in accs],
            marker=marker, color=color, linewidth=1.8,
            markersize=6, label=name)
    ax.set_xlabel("SNR(dB)", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(f"{name} — RADIOML 2016.10A", fontsize=11)
    ax.set_xticks(SNRS)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}%"))
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fname = f"{name}_accuracy.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {fname}")

# Combined 2D classifiers plot
fig, ax = plt.subplots(figsize=(7, 5))
for name, _, _, color, marker in MODELS:
    ax.plot(SNRS, [a*100 for a in all_accs[name]],
            marker=marker, color=color, linewidth=1.8,
            markersize=6, label=name)
ax.set_xlabel("SNR(dB)", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("2D Classifiers — RADIOML 2016.10A", fontsize=11)
ax.set_xticks(SNRS)
ax.tick_params(axis='x', labelsize=8)
ax.set_ylim(0, 100)
ax.set_yticks(range(0, 101, 10))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}%"))
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("2D_Classifiers_combined_accuracy.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved → 2D_Classifiers_combined_accuracy.png")

print("\nAll 2D classifier results:")
for name, accs in all_accs.items():
    print(f"\n{name}:")
    for s, a in zip(SNRS, accs):
        print(f"  SNR {s:+4d} dB  →  {a*100:.2f}%")

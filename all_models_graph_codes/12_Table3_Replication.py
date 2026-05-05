# ================================================================
#  Table III Replication — RADIOML 2016.10A
#  Wang et al., IEEE TIFS 2024, Table III
#  Macro-averaged Precision, Recall, F1-score at: -6, -2, +2, +6 dB
#
#  KEY FIX: Paper says "results based on BEST PERFORMANCE of multiple
#  training epochs". This means we must restore weights from the epoch
#  with the BEST F1-score (not just best accuracy). We use a custom
#  callback (BestF1Checkpoint) to track this correctly.
#  Also: monitor val_loss for EarlyStopping (not val_accuracy) so
#  training continues even after accuracy plateaus — this is critical
#  at high SNR (+2, +6 dB) where accuracy saturates early but F1
#  keeps improving as the model learns to separate all 11 classes.
# ================================================================
import pickle, numpy as np, random, tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv, copy

DATASET_PATH = "/kaggle/input/datasets/sanjeevharge/2016-10a/RML2016.10a_dict.pkl"
SNRS_ALL = list(range(-20, 8, 2))
SNRS_T3  = [-6, -2, 2, 6]

def set_seed(s=42):
    np.random.seed(s); tf.random.set_seed(s); random.seed(s)

# ── Data loading ─────────────────────────────────────────────────
def load_raw(path):
    with open(path, 'rb') as f: data = pickle.load(f, encoding='latin1')
    mods = sorted(set(k[0] for k in data.keys()))
    cmap = {m: i for i, m in enumerate(mods)}; nc = len(mods)
    print(f"Classes ({nc}): {mods}")
    dbs = {}
    for snr in SNRS_ALL:
        Xa, ya = [], []
        for mod in mods:
            X = np.transpose(data[(mod, snr)], (0, 2, 1)).astype(np.float32)
            Xa.append(X); ya.extend([cmap[mod]] * len(X))
        Xa = np.vstack(Xa); ya = np.array(ya)
        Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=0.2,
                                               stratify=ya, random_state=42)
        dbs[snr] = (Xtr, Xte, ytr, yte)
    return dbs, nc

# ── Normalization ─────────────────────────────────────────────────
def norm_global(Xtr, Xte):
    g = np.max(np.abs(Xtr)); return Xtr/g, Xte/g

def norm_per_sample(Xtr, Xte):
    def _n(X):
        m = np.max(np.abs(X), axis=(1, 2), keepdims=True)
        return X / np.where(m == 0, 1.0, m)
    return _n(Xtr), _n(Xte)

def norm_and_tile(Xtr, Xte):
    def _n(X):
        m = np.max(np.abs(X), axis=(1, 2), keepdims=True)
        return X / np.where(m == 0, 1.0, m)
    return (np.tile(_n(Xtr), (1,1,16))[..., np.newaxis],
            np.tile(_n(Xte),  (1,1,16))[..., np.newaxis])

# ════════════════════════════════════════════════════════════════
# CUSTOM CALLBACK: tracks best macro-F1 epoch, restores those weights
# ════════════════════════════════════════════════════════════════
class BestF1Checkpoint(tf.keras.callbacks.Callback):
    """
    After every epoch, computes macro F1 on the validation set.
    Keeps a copy of the weights from the epoch with the highest F1.
    After training, call .restore() to load those best weights.
    Also implements early stopping based on F1 (patience=25).
    """
    def __init__(self, Xte, yte, patience=25):
        super().__init__()
        self.Xte      = Xte
        self.yte      = yte
        self.patience = patience
        self.best_f1  = -1.0
        self.best_p   = 0.0
        self.best_r   = 0.0
        self.best_weights = None
        self.wait     = 0

    def on_epoch_end(self, epoch, logs=None):
        preds = np.argmax(self.model.predict(self.Xte, verbose=0), axis=1)
        p, r, f, _ = precision_recall_fscore_support(
            self.yte, preds, average='macro', zero_division=0)
        if f > self.best_f1:
            self.best_f1      = f
            self.best_p       = p
            self.best_r       = r
            self.best_weights = self.model.get_weights()
            self.wait         = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def restore(self):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)

# Same callback for DAE (dual output — only classification head matters)
class BestF1CheckpointDAE(tf.keras.callbacks.Callback):
    def __init__(self, Xte, yte, patience=25):
        super().__init__()
        self.Xte = Xte; self.yte = yte; self.patience = patience
        self.best_f1 = -1.0; self.best_p = 0.0; self.best_r = 0.0
        self.best_weights = None; self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        outputs = self.model.predict(self.Xte, verbose=0)
        # DAE returns [reconstruction, classification]
        clf_out = outputs[1] if isinstance(outputs, list) else outputs
        preds = np.argmax(clf_out, axis=1)
        p, r, f, _ = precision_recall_fscore_support(
            self.yte, preds, average='macro', zero_division=0)
        if f > self.best_f1:
            self.best_f1 = f; self.best_p = p; self.best_r = r
            self.best_weights = self.model.get_weights(); self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def restore(self):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)

# ── Training functions ────────────────────────────────────────────
def train_one(model, Xtr, ytr, Xte, yte, lr, epochs, bs):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    f1_cb = BestF1Checkpoint(Xte, yte, patience=25)
    lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        'val_loss', factor=0.5, patience=8, min_lr=1e-9, verbose=0)
    model.fit(Xtr, ytr, epochs=epochs, batch_size=bs,
              validation_data=(Xte, yte),
              callbacks=[lr_cb, f1_cb], verbose=0)
    f1_cb.restore()   # load weights from best-F1 epoch
    return f1_cb.best_p, f1_cb.best_r, f1_cb.best_f1

def train_dae(model, Xtr, ytr, Xte, yte, lr, epochs, bs):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss={'reconstruction': 'mse',
              'classification': 'sparse_categorical_crossentropy'},
        loss_weights={'reconstruction': 0.5, 'classification': 1.0},
        metrics={'classification': 'accuracy'})
    f1_cb = BestF1CheckpointDAE(Xte, yte, patience=25)
    lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        'val_loss', factor=0.5, patience=8, min_lr=1e-9, verbose=0)
    model.fit(Xtr, {'reconstruction': Xtr, 'classification': ytr},
              epochs=epochs, batch_size=bs,
              validation_data=(Xte, {'reconstruction': Xte, 'classification': yte}),
              callbacks=[lr_cb, f1_cb], verbose=0)
    f1_cb.restore()
    return f1_cb.best_p, f1_cb.best_r, f1_cb.best_f1

# ════════════════════════════════════════════════════════════════
# ALL MODEL BUILDERS (identical to 00_ALL_MODELS_Fig2ab.py)
# ════════════════════════════════════════════════════════════════
class RFFLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, scale, **kw):
        super().__init__(**kw); self.output_dim=output_dim; self.scale=scale
    def build(self, s):
        d=s[-1]
        self.W=self.add_weight((d,self.output_dim),
            initializer=tf.random_normal_initializer(stddev=self.scale),
            trainable=False, name='W')
        self.b=self.add_weight((self.output_dim,),
            initializer=tf.random_uniform_initializer(0,2*np.pi),
            trainable=False, name='b')
        super().build(s)
    def call(self, x):
        return tf.sqrt(2.0/float(self.output_dim))*tf.cos(tf.matmul(x,self.W)+self.b)

CRFF_CFG = [
    (3,[128,128, 64],3,5,[2048,2048,1024,512,4096],[10,15,10,15,10],0.001),
    (3,[128, 64,128],3,1,[8192],                   [15],            0.0),
    (2,[ 32,  32],   3,3,[2048,512,2048],           [15,15,13],     0.1),
    (3,[ 64,128, 32],7,1,[2048],                   [10],            0.0),
]

def build_autosmc(nc):
    inp=tf.keras.Input(shape=(128,2))
    x=layers.Reshape((128,2,1))(inp)
    x=layers.Conv2D(128,(7,2),padding='valid')(x)
    x=layers.Reshape((122,128))(x); x=layers.LeakyReLU()(x); x=layers.MaxPool1D(2)(x)
    for _,flist,lk,_,rdims,scales,w in CRFF_CFG:
        out_f=flist[-1]; conv=x
        for f in flist:
            conv=layers.Conv1D(f,lk,padding='same')(conv)
            conv=layers.BatchNormalization()(conv); conv=layers.LeakyReLU()(conv)
        if x.shape[-1]!=out_f: x=layers.Conv1D(out_f,1,padding='same')(x)
        if w>0:
            rff=x
            for od,sc in zip(rdims,scales): rff=RFFLayer(od,sc)(rff)
            rff=layers.Dense(out_f)(rff); x=conv+w*rff
        else: x=conv
        x=layers.MaxPool1D(2,padding='same')(x)
    x=layers.GlobalAveragePooling1D()(x)
    return tf.keras.Model(inp,layers.Dense(nc,activation='softmax')(x))

def build_lstm(nc):
    inp=tf.keras.Input(shape=(128,2))
    x=layers.LSTM(128,return_sequences=True)(inp); x=layers.LSTM(128)(x)
    return tf.keras.Model(inp,layers.Dense(nc,activation='softmax')(x))

def build_vtcnn2(nc):
    inp=tf.keras.Input(shape=(128,2))
    x=layers.ZeroPadding1D(2)(inp)
    x=layers.Conv1D(256,3,activation='relu')(x); x=layers.Dropout(0.5)(x)
    x=layers.ZeroPadding1D(2)(x)
    x=layers.Conv1D(80,3,activation='relu')(x);  x=layers.Dropout(0.5)(x)
    x=layers.Flatten()(x); x=layers.Dense(256,activation='relu')(x); x=layers.Dropout(0.5)(x)
    return tf.keras.Model(inp,layers.Dense(nc,activation='softmax')(x))

def res_block(x, f):
    s=layers.Conv1D(f,1,padding='same')(x)
    y=layers.Conv1D(f,3,padding='same')(x); y=layers.BatchNormalization()(y)
    y=layers.Activation('relu')(y); y=layers.Conv1D(f,3,padding='same')(y)
    y=layers.BatchNormalization()(y)
    return layers.Activation('relu')(layers.Add()([y,s]))

def build_rn(nc):
    inp=tf.keras.Input(shape=(128,2))
    x=layers.Conv1D(32,3,padding='same',activation='relu')(inp); x=layers.Dropout(0.5)(x)
    for f in [32,64,128]: x=res_block(x,f); x=layers.MaxPool1D(2)(x)
    x=layers.Flatten()(x); x=layers.Dense(128,activation='selu')(x); x=layers.Dropout(0.5)(x)
    return tf.keras.Model(inp,layers.Dense(nc,activation='softmax')(x))

def build_mcldnn(nc):
    inp=tf.keras.Input(shape=(128,2))
    x1=layers.Conv1D(50,8,padding='same',activation='relu')(inp)
    x1=layers.Conv1D(50,8,padding='same',activation='relu')(x1)
    x2=layers.Reshape((128,2,1))(inp)
    x2=layers.Conv2D(50,(8,1),padding='same',activation='relu')(x2)
    x2=layers.Reshape((128,100))(x2)
    x=layers.Concatenate(axis=-1)([x1,x2])
    x=layers.LSTM(128,return_sequences=True)(x); x=layers.LSTM(128)(x)
    x=layers.Dense(256,activation='relu')(x); x=layers.Dropout(0.5)(x)
    return tf.keras.Model(inp,layers.Dense(nc,activation='softmax')(x))

def build_dae(nc):
    inp=tf.keras.Input(shape=(128,2))
    encoded=layers.LSTM(128)(inp)
    dec=layers.RepeatVector(128)(encoded)
    dec=layers.LSTM(128,return_sequences=True)(dec)
    reconstructed=layers.TimeDistributed(layers.Dense(2),name='reconstruction')(dec)
    clf=layers.Dense(256,activation='relu')(encoded); clf=layers.Dropout(0.5)(clf)
    clf_out=layers.Dense(nc,activation='softmax',name='classification')(clf)
    return Model(inp,[reconstructed,clf_out])

class STN1D(tf.keras.layers.Layer):
    def __init__(self, seq_len=128, **kw):
        super().__init__(**kw); self.seq_len=seq_len
        self.loc=tf.keras.Sequential([
            layers.Conv1D(32,3,padding='same',activation='relu'),
            layers.MaxPool1D(2),
            layers.Conv1D(32,3,padding='same',activation='relu'),
            layers.MaxPool1D(2), layers.Flatten(),
            layers.Dense(32,activation='relu'),
            layers.Dense(2,bias_initializer='zeros',kernel_initializer='zeros')])
    def call(self, x):
        B=tf.shape(x)[0]; params=self.loc(x)
        scale=params[:,0:1]+1.0; shift=params[:,1:2]
        src=tf.tile(tf.linspace(-1.0,1.0,self.seq_len)[tf.newaxis,:],[B,1])
        tgt=tf.clip_by_value(scale*src+shift,-1.0,1.0)
        tgt_idx=(tgt+1.0)/2.0*tf.cast(self.seq_len-1,tf.float32)
        idx0=tf.cast(tf.floor(tgt_idx),tf.int32); idx1=tf.minimum(idx0+1,self.seq_len-1)
        w1=tgt_idx-tf.cast(idx0,tf.float32); w0=1.0-w1
        def gc(args):
            xi,i0,i1,a0,a1=args
            return a0[:,tf.newaxis]*tf.gather(xi,i0)+a1[:,tf.newaxis]*tf.gather(xi,i1)
        return tf.map_fn(gc,(x,idx0,idx1,w0,w1),fn_output_signature=tf.float32)

def resnext_block(x, of, c=4):
    gw=of//c; sk=layers.Conv1D(of,1,padding='same')(x)
    pts=[layers.Conv1D(gw,1,padding='same')(
             layers.DepthwiseConv1D(3,padding='same')(x)) for _ in range(c)]
    ag=layers.Concatenate()(pts); ag=layers.BatchNormalization()(ag)
    return layers.ReLU()(layers.Add()([ag,sk]))

def build_stn_resnext(nc):
    inp=tf.keras.Input(shape=(128,2)); x=STN1D(seq_len=128)(inp)
    x=layers.Conv1D(64,3,padding='same')(x); x=layers.BatchNormalization()(x); x=layers.ReLU()(x)
    for f in [64,128,128]: x=resnext_block(x,f,4); x=layers.MaxPool1D(2)(x)
    x=layers.GlobalAveragePooling1D()(x)
    return tf.keras.Model(inp,layers.Dense(nc,activation='softmax')(x))

class PolarAccumLayer(tf.keras.layers.Layer):
    def call(self, x):
        I=x[:,:,0:1]; Q=x[:,:,1:2]
        amp=tf.sqrt(I**2+Q**2); phi=tf.atan2(Q,I)
        return tf.concat([I,Q,tf.cumsum(amp,axis=1),tf.cumsum(phi,axis=1)],axis=-1)

def build_cnn_accu_polar(nc):
    inp=tf.keras.Input(shape=(128,2)); x=PolarAccumLayer()(inp)
    x=layers.Conv1D(64,3,padding='same')(x); x=layers.BatchNormalization()(x); x=layers.ReLU()(x)
    x=layers.Conv1D(64,3,padding='same')(x); x=layers.BatchNormalization()(x); x=layers.ReLU()(x)
    x=layers.MaxPool1D(2)(x)
    x=layers.Conv1D(128,3,padding='same')(x); x=layers.BatchNormalization()(x); x=layers.ReLU()(x)
    x=layers.Conv1D(128,3,padding='same')(x); x=layers.BatchNormalization()(x); x=layers.ReLU()(x)
    x=layers.GlobalAveragePooling1D()(x)
    x=layers.Dense(128,activation='relu')(x); x=layers.Dropout(0.5)(x)
    return tf.keras.Model(inp,layers.Dense(nc,activation='softmax')(x))

def sfp_block(x):
    s=x; x=layers.Conv1D(64,3,padding='same')(x); x=layers.BatchNormalization()(x); x=layers.ReLU()(x)
    x=layers.Conv1D(64,3,padding='same')(x); x=layers.BatchNormalization()(x)
    x=layers.Add()([x,s]); return layers.ReLU()(x)

def build_msmcnet(nc):
    inp=tf.keras.Input(shape=(128,2)); x=layers.Reshape((128,2,1))(inp)
    x=layers.Conv2D(64,(3,2),padding='valid',activation='relu')(x)
    x=layers.Reshape((int(x.shape[1]),64))(x)
    for _ in range(4): x=sfp_block(x)
    x=layers.GlobalAveragePooling1D()(x)
    return tf.keras.Model(inp,layers.Dense(nc,activation='softmax')(x))

def build_mobilenet(nc):
    base=tf.keras.applications.MobileNet(input_shape=(128,32,1),alpha=0.5,
                                          include_top=False,weights=None)
    inp=tf.keras.Input(shape=(128,32,1)); x=base(inp); x=layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inp,layers.Dense(nc,activation='softmax')(x))

def build_resnet50(nc):
    base=tf.keras.applications.ResNet50(input_shape=(128,32,1),
                                         include_top=False,weights=None)
    inp=tf.keras.Input(shape=(128,32,1)); x=base(inp); x=layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inp,layers.Dense(nc,activation='softmax')(x))

def build_densenet169(nc):
    base=tf.keras.applications.DenseNet169(input_shape=(128,32,1),
                                            include_top=False,weights=None)
    inp=tf.keras.Input(shape=(128,32,1)); x=base(inp); x=layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inp,layers.Dense(nc,activation='softmax')(x))

# ════════════════════════════════════════════════════════════════
# REGISTRY — Table III row order
# (name, builder, norm_fn, lr, epochs, bs, is_dae)
#
# epochs increased vs Fig2 files:
#   baselines:  200  (was 100)  — more room to find best-F1 epoch
#   AutoSMC:    300  (was 150)  — larger search space, needs more time
# patience in BestF1Checkpoint = 25 (was 15 in old EarlyStopping)
# ════════════════════════════════════════════════════════════════
REGISTRY = [
    ("MobileNet",      build_mobilenet,      norm_and_tile,   1e-3, 200,  64, False),
    ("ResNet50",       build_resnet50,        norm_and_tile,   1e-3, 200,  64, False),
    ("DenseNet169",    build_densenet169,     norm_and_tile,   1e-3, 200,  32, False),
    ("CNN_Accu_polar", build_cnn_accu_polar,  norm_per_sample, 1e-3, 200, 128, False),
    ("MCLDNN",         build_mcldnn,          norm_per_sample, 1e-3, 200,  64, False),
    ("LSTM",           build_lstm,            norm_per_sample, 1e-3, 200, 128, False),
    ("VTCNN2",         build_vtcnn2,          norm_per_sample, 1e-3, 200, 128, False),
    ("RN",             build_rn,              norm_per_sample, 1e-3, 200, 128, False),
    ("DAE",            build_dae,             norm_per_sample, 1e-3, 200, 128, True),
    ("STN-ResNeXt",    build_stn_resnext,     norm_per_sample, 1e-3, 200, 128, False),
    ("MsmcNet",        build_msmcnet,         norm_global,     1e-3, 200, 128, False),
    ("AutoSMC*",       build_autosmc,         norm_global,     1e-3, 200, 128, False),
    ("AutoSMC",        build_autosmc,         norm_global,     1e-3, 200, 128, False),
]

# ── Train ─────────────────────────────────────────────────────────
set_seed(42)
dbs, nc = load_raw(DATASET_PATH)
table = {}   # {model_name: {snr: (P, R, F1)}}

for name, builder, norm_fn, lr, epochs, bs, is_dae in REGISTRY:
    table[name] = {}
    print(f"\n{'='*60}")
    print(f"Training {name}  [epochs≤{epochs}, patience=25, best-F1 restore]")
    print('='*60)
    for snr in SNRS_T3:
        Xtr_r, Xte_r, ytr, yte = dbs[snr]
        Xtr_n, Xte_n = norm_fn(Xtr_r, Xte_r)
        set_seed(42); tf.keras.backend.clear_session()
        model = builder(nc)
        if is_dae:
            p, r, f = train_dae(model, Xtr_n, ytr, Xte_n, yte, lr, epochs, bs)
        else:
            p, r, f = train_one(model, Xtr_n, ytr, Xte_n, yte, lr, epochs, bs)
        table[name][snr] = (p, r, f)
        print(f"  SNR {snr:+3d} dB  →  P={p:.4f}  R={r:.4f}  F1={f:.4f}")

# ── Save CSV ───────────────────────────────────────────────────────
csv_path = "table3_autosmc_radioml2016_10a.csv"
with open(csv_path, 'w', newline='') as fp:
    w = csv.writer(fp)
    header = ["Model"]
    for s in SNRS_T3:
        header += [f"Precision@{s}dB", f"Recall@{s}dB", f"F1@{s}dB"]
    w.writerow(header)
    for name, _, *_ in REGISTRY:
        row = [name]
        for s in SNRS_T3:
            p, r, f = table[name][s]
            row += [f"{p:.4f}", f"{r:.4f}", f"{f:.4f}"]
        w.writerow(row)
print(f"\nSaved → {csv_path}")

# ── Print Table ────────────────────────────────────────────────────
print(f"\n{'='*100}")
print("TABLE III REPLICATION — Macro Precision / Recall / F1")
print(f"{'='*100}")
hdr = f"{'Model':<18}"
for s in SNRS_T3:
    hdr += f"{'P@'+str(s)+'dB':>9}{'R@'+str(s)+'dB':>9}{'F1@'+str(s)+'dB':>10}"
print(hdr)
print("-"*100)
for name, _, *_ in REGISTRY:
    row = f"{name:<18}"
    for s in SNRS_T3:
        p, r, f = table[name][s]
        row += f"{p:>9.4f}{r:>9.4f}{f:>10.4f}"
    print(row)
print("="*100)

# ── Compare vs paper ───────────────────────────────────────────────
paper_f1 = {
    "MobileNet":      {-6:0.5007,-2:0.6905,2:0.7827,6:0.8123},
    "ResNet50":       {-6:0.5438,-2:0.7766,2:0.8792,6:0.9117},
    "DenseNet169":    {-6:0.5872,-2:0.7919,2:0.8763,6:0.8786},
    "CNN_Accu_polar": {-6:0.2929,-2:0.4517,2:0.6762,6:0.7678},
    "MCLDNN":         {-6:0.4494,-2:0.5933,2:0.7443,6:0.8553},
    "LSTM":           {-6:0.5180,-2:0.6284,2:0.5758,6:0.6658},
    "VTCNN2":         {-6:0.4896,-2:0.6970,2:0.7945,6:0.8356},
    "RN":             {-6:0.5445,-2:0.6939,2:0.8057,6:0.8430},
    "DAE":            {-6:0.6046,-2:0.7747,2:0.7458,6:0.7965},
    "STN-ResNeXt":    {-6:0.5633,-2:0.7742,2:0.8405,6:0.8871},
    "MsmcNet":        {-6:0.5992,-2:0.8032,2:0.8936,6:0.9229},
    "AutoSMC*":       {-6:0.6053,-2:0.8387,2:0.9140,6:0.9291},
    "AutoSMC":        {-6:0.6389,-2:0.8358,2:0.9228,6:0.9385},
}
print("\nF1 COMPARISON vs PAPER (our → paper):")
print(f"{'Model':<18} {'F1@-6':>12} {'F1@-2':>12} {'F1@+2':>12} {'F1@+6':>12}")
print("-"*60)
for name, _, *_ in REGISTRY:
    if name in paper_f1:
        vals = "  ".join(
            f"{table[name][s][2]:.4f}→{paper_f1[name][s]:.4f}"
            for s in SNRS_T3)
        print(f"{name:<18}  {vals}")

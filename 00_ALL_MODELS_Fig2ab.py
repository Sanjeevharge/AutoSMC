# ================================================================
#  ALL MODELS — RADIOML 2016.10A
#  Reproduces Fig 2(a) and Fig 2(b) from Wang et al., IEEE TIFS 2024
#
#  Fig 2(a): MobileNet, ResNet50, DenseNet169, AutoSMC*, AutoSMC
#  Fig 2(b): CNN_Accu_polar, MCLDNN, LSTM, VTCNN2, RN, DAE,
#             STN-ResNeXt, MsmcNet, AutoSMC*, AutoSMC
#
#  Normalization:
#    global-max   → AutoSMC, AutoSMC*, MsmcNet
#    per-sample   → all other 1D models
#    per-sample+tile16x → MobileNet, ResNet50, DenseNet169
# ================================================================
import pickle, numpy as np, random, tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATASET_PATH = "/kaggle/input/datasets/sanjeevharge/2016-10a/RML2016.10a_dict.pkl"
SNRS = list(range(-20, 8, 2))

# ── Reproducibility ─────────────────────────────────────────────
def set_seed(s=42):
    np.random.seed(s); tf.random.set_seed(s); random.seed(s)

# ── Data loading ────────────────────────────────────────────────
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

# ── Normalization ───────────────────────────────────────────────
def norm_global(Xtr, Xte):
    g = np.max(np.abs(Xtr)); return Xtr/g, Xte/g

def norm_per_sample(Xtr, Xte):
    def _n(X):
        m = np.max(np.abs(X), axis=(1,2), keepdims=True)
        return X / np.where(m==0, 1.0, m)
    return _n(Xtr), _n(Xte)

def norm_and_tile(Xtr, Xte):
    def _n(X):
        m = np.max(np.abs(X), axis=(1,2), keepdims=True)
        return X / np.where(m==0, 1.0, m)
    def _tile(X): return np.tile(_n(X), (1,1,16))[..., np.newaxis]
    return _tile(Xtr), _tile(Xte)

# ── Training loop ───────────────────────────────────────────────
def train_one(model, Xtr, ytr, Xte, yte, lr, epochs, bs):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cb = [
        tf.keras.callbacks.ReduceLROnPlateau('val_loss',0.5,5,min_lr=1e-9,verbose=0),
        tf.keras.callbacks.EarlyStopping('val_accuracy',patience=15,
                                          restore_best_weights=True,verbose=0),
    ]
    model.fit(Xtr, ytr, epochs=epochs, batch_size=bs,
              validation_data=(Xte,yte), callbacks=cb, verbose=0)
    return float(accuracy_score(yte, np.argmax(model.predict(Xte,verbose=0),axis=1)))

def train_dae(model, Xtr, ytr, Xte, yte, lr, epochs, bs):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss={'reconstruction':'mse',
                        'classification':'sparse_categorical_crossentropy'},
                  loss_weights={'reconstruction':0.5,'classification':1.0},
                  metrics={'classification':'accuracy'})
    cb = [
        tf.keras.callbacks.ReduceLROnPlateau('val_loss',0.5,5,min_lr=1e-9,verbose=0),
        tf.keras.callbacks.EarlyStopping('val_classification_accuracy',patience=15,
                                          restore_best_weights=True,verbose=0),
    ]
    model.fit(Xtr, {'reconstruction':Xtr,'classification':ytr},
              epochs=epochs, batch_size=bs,
              validation_data=(Xte,{'reconstruction':Xte,'classification':yte}),
              callbacks=cb, verbose=0)
    _, clf_out = model.predict(Xte, verbose=0)
    return float(accuracy_score(yte, np.argmax(clf_out,axis=1)))

# ════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ════════════════════════════════════════════════════════════════

# ── AutoSMC / AutoSMC* (Table V architecture) ──────────────────
class RFFLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, scale, **kw):
        super().__init__(**kw); self.output_dim=output_dim; self.scale=scale
    def build(self, s):
        d=s[-1]
        self.W=self.add_weight((d,self.output_dim),
            initializer=tf.random_normal_initializer(stddev=self.scale),
            trainable=False,name='W')
        self.b=self.add_weight((self.output_dim,),
            initializer=tf.random_uniform_initializer(0,2*np.pi),
            trainable=False,name='b')
        super().build(s)
    def call(self,x):
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
    x=layers.Reshape((122,128))(x)
    x=layers.LeakyReLU()(x); x=layers.MaxPool1D(2)(x)
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

# ── LSTM ────────────────────────────────────────────────────────
def build_lstm(nc):
    inp=tf.keras.Input(shape=(128,2))
    x=layers.LSTM(128,return_sequences=True)(inp); x=layers.LSTM(128)(x)
    return tf.keras.Model(inp,layers.Dense(nc,activation='softmax')(x))

# ── VTCNN2 ──────────────────────────────────────────────────────
def build_vtcnn2(nc):
    inp=tf.keras.Input(shape=(128,2))
    x=layers.ZeroPadding1D(2)(inp)
    x=layers.Conv1D(256,3,activation='relu')(x); x=layers.Dropout(0.5)(x)
    x=layers.ZeroPadding1D(2)(x)
    x=layers.Conv1D(80,3,activation='relu')(x);  x=layers.Dropout(0.5)(x)
    x=layers.Flatten()(x)
    x=layers.Dense(256,activation='relu')(x); x=layers.Dropout(0.5)(x)
    return tf.keras.Model(inp,layers.Dense(nc,activation='softmax')(x))

# ── RN ──────────────────────────────────────────────────────────
def res_block(x,f):
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

# ── MCLDNN ──────────────────────────────────────────────────────
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

# ── DAE (joint autoencoder + classifier) ────────────────────────
def build_dae(nc):
    inp=tf.keras.Input(shape=(128,2))
    encoded=layers.LSTM(128)(inp)
    dec=layers.RepeatVector(128)(encoded)
    dec=layers.LSTM(128,return_sequences=True)(dec)
    reconstructed=layers.TimeDistributed(layers.Dense(2),name='reconstruction')(dec)
    clf=layers.Dense(256,activation='relu')(encoded); clf=layers.Dropout(0.5)(clf)
    clf_out=layers.Dense(nc,activation='softmax',name='classification')(clf)
    return Model(inp,[reconstructed,clf_out])

# ── STN-ResNeXt ─────────────────────────────────────────────────
class STN1D(tf.keras.layers.Layer):
    def __init__(self,seq_len=128,**kw):
        super().__init__(**kw); self.seq_len=seq_len
        self.loc=tf.keras.Sequential([
            layers.Conv1D(32,3,padding='same',activation='relu'),
            layers.MaxPool1D(2),
            layers.Conv1D(32,3,padding='same',activation='relu'),
            layers.MaxPool1D(2), layers.Flatten(),
            layers.Dense(32,activation='relu'),
            layers.Dense(2,bias_initializer='zeros',kernel_initializer='zeros')])
    def call(self,x):
        B=tf.shape(x)[0]
        params=self.loc(x); scale=params[:,0:1]+1.0; shift=params[:,1:2]
        src=tf.linspace(-1.0,1.0,self.seq_len)
        src=tf.tile(src[tf.newaxis,:],[B,1])
        tgt=tf.clip_by_value(scale*src+shift,-1.0,1.0)
        tgt_idx=(tgt+1.0)/2.0*tf.cast(self.seq_len-1,tf.float32)
        idx0=tf.cast(tf.floor(tgt_idx),tf.int32)
        idx1=tf.minimum(idx0+1,self.seq_len-1)
        w1=tgt_idx-tf.cast(idx0,tf.float32); w0=1.0-w1
        def gather_ch(args):
            xi,i0,i1,a0,a1=args
            return a0[:,tf.newaxis]*tf.gather(xi,i0)+a1[:,tf.newaxis]*tf.gather(xi,i1)
        return tf.map_fn(gather_ch,(x,idx0,idx1,w0,w1),fn_output_signature=tf.float32)

def resnext_block(x,out_f,cardinality=4):
    gw=out_f//cardinality; skip=layers.Conv1D(out_f,1,padding='same')(x)
    paths=[layers.Conv1D(gw,1,padding='same')(
               layers.DepthwiseConv1D(3,padding='same')(x)) for _ in range(cardinality)]
    agg=layers.Concatenate()(paths); agg=layers.BatchNormalization()(agg)
    return layers.ReLU()(layers.Add()([agg,skip]))

def build_stn_resnext(nc):
    inp=tf.keras.Input(shape=(128,2))
    x=STN1D(seq_len=128)(inp)
    x=layers.Conv1D(64,3,padding='same')(x); x=layers.BatchNormalization()(x); x=layers.ReLU()(x)
    for f in [64,128,128]: x=resnext_block(x,f,4); x=layers.MaxPool1D(2)(x)
    x=layers.GlobalAveragePooling1D()(x)
    return tf.keras.Model(inp,layers.Dense(nc,activation='softmax')(x))

# ── CNNAccu_polar ───────────────────────────────────────────────
class PolarAccumLayer(tf.keras.layers.Layer):
    def call(self,x):
        I=x[:,:,0:1]; Q=x[:,:,1:2]
        amp=tf.sqrt(I**2+Q**2); phi=tf.atan2(Q,I)
        return tf.concat([I,Q,tf.cumsum(amp,axis=1),tf.cumsum(phi,axis=1)],axis=-1)

def build_cnn_accu_polar(nc):
    inp=tf.keras.Input(shape=(128,2))
    x=PolarAccumLayer()(inp)
    x=layers.Conv1D(64,3,padding='same')(x); x=layers.BatchNormalization()(x); x=layers.ReLU()(x)
    x=layers.Conv1D(64,3,padding='same')(x); x=layers.BatchNormalization()(x); x=layers.ReLU()(x)
    x=layers.MaxPool1D(2)(x)
    x=layers.Conv1D(128,3,padding='same')(x); x=layers.BatchNormalization()(x); x=layers.ReLU()(x)
    x=layers.Conv1D(128,3,padding='same')(x); x=layers.BatchNormalization()(x); x=layers.ReLU()(x)
    x=layers.GlobalAveragePooling1D()(x)
    x=layers.Dense(128,activation='relu')(x); x=layers.Dropout(0.5)(x)
    return tf.keras.Model(inp,layers.Dense(nc,activation='softmax')(x))

# ── MsmcNet ─────────────────────────────────────────────────────
def sfp_block(x):
    s=x; x=layers.Conv1D(64,3,padding='same')(x); x=layers.BatchNormalization()(x)
    x=layers.ReLU()(x); x=layers.Conv1D(64,3,padding='same')(x)
    x=layers.BatchNormalization()(x); x=layers.Add()([x,s]); return layers.ReLU()(x)

def build_msmcnet(nc):
    inp=tf.keras.Input(shape=(128,2))
    x=layers.Reshape((128,2,1))(inp)
    x=layers.Conv2D(64,(3,2),padding='valid',activation='relu')(x)
    x=layers.Reshape((int(x.shape[1]),64))(x)
    for _ in range(4): x=sfp_block(x)
    x=layers.GlobalAveragePooling1D()(x)
    return tf.keras.Model(inp,layers.Dense(nc,activation='softmax')(x))

# ── 2D models ───────────────────────────────────────────────────
def build_mobilenet(nc):
    base=tf.keras.applications.MobileNet(input_shape=(128,32,1),alpha=0.5,
                                          include_top=False,weights=None)
    inp=tf.keras.Input(shape=(128,32,1)); x=base(inp)
    x=layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inp,layers.Dense(nc,activation='softmax')(x))

def build_resnet50(nc):
    base=tf.keras.applications.ResNet50(input_shape=(128,32,1),
                                         include_top=False,weights=None)
    inp=tf.keras.Input(shape=(128,32,1)); x=base(inp)
    x=layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inp,layers.Dense(nc,activation='softmax')(x))

def build_densenet169(nc):
    base=tf.keras.applications.DenseNet169(input_shape=(128,32,1),
                                            include_top=False,weights=None)
    inp=tf.keras.Input(shape=(128,32,1)); x=base(inp)
    x=layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inp,layers.Dense(nc,activation='softmax')(x))

# ════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# (name, builder, norm_fn, lr, epochs, bs, is_dae, is_2d)
# ════════════════════════════════════════════════════════════════
REGISTRY = [
    ("MobileNet",      build_mobilenet,      norm_and_tile,    1e-3, 100,  64, False, True),
    ("ResNet50",       build_resnet50,       norm_and_tile,    1e-3, 100,  64, False, True),
    ("DenseNet169",    build_densenet169,    norm_and_tile,    1e-3, 100,  32, False, True),
    ("AutoSMC*",       build_autosmc,        norm_global,      1e-3, 200, 128, False, False),
    ("AutoSMC",        build_autosmc,        norm_global,      1e-3, 200, 128, False, False),
    ("CNN_Accu_polar", build_cnn_accu_polar, norm_per_sample,  1e-3, 100, 128, False, False),
    ("MCLDNN",         build_mcldnn,         norm_per_sample,  1e-3, 100,  64, False, False),
    ("LSTM",           build_lstm,           norm_per_sample,  1e-3, 100, 128, False, False),
    ("VTCNN2",         build_vtcnn2,         norm_per_sample,  1e-3, 100, 128, False, False),
    ("RN",             build_rn,             norm_per_sample,  1e-3, 100, 128, False, False),
    ("DAE",            build_dae,            norm_per_sample,  1e-3, 100, 128, True,  False),
    ("STN-ResNeXt",    build_stn_resnext,    norm_per_sample,  1e-3, 100, 128, False, False),
    ("MsmcNet",        build_msmcnet,        norm_global,      1e-3, 100, 128, False, False),
]

# ── Plot style (paper Fig 2 colours and markers) ────────────────
STYLE = {
    "MobileNet":      ('#b5a400', 's'),
    "ResNet50":       ('#ff7f0e', '+'),
    "DenseNet169":    ('#2ca02c', 'D'),
    "AutoSMC*":       ('#c4a0c0', 'v'),
    "AutoSMC":        ('#c00040', 's'),
    "CNN_Accu_polar": ('#e07820', 'o'),
    "MCLDNN":         ('#ff7f0e', '^'),
    "LSTM":           ('#1a1a1a', 's'),
    "VTCNN2":         ('#6baed6', 'x'),
    "RN":             ('#54278f', 'D'),
    "DAE":            ('#6baed6', 'o'),
    "STN-ResNeXt":    ('#e6c619', 'h'),
    "MsmcNet":        ('#545454', 'P'),
}

# ── Train all models ────────────────────────────────────────────
set_seed(42)
dbs, nc = load_raw(DATASET_PATH)
results = {}

for name, builder, norm_fn, lr, epochs, bs, is_dae, is_2d in REGISTRY:
    accs = []
    print(f"\n{'='*60}")
    print(f"Training {name}")
    print('='*60)
    for snr in SNRS:
        Xtr_r, Xte_r, ytr, yte = dbs[snr]
        Xtr_n, Xte_n = norm_fn(Xtr_r, Xte_r)
        set_seed(42); tf.keras.backend.clear_session()
        model = builder(nc)
        if is_dae:
            acc = train_dae(model, Xtr_n, ytr, Xte_n, yte, lr, epochs, bs)
        else:
            acc = train_one(model, Xtr_n, ytr, Xte_n, yte, lr, epochs, bs)
        accs.append(acc)
        print(f"  SNR {snr:+4d} dB  →  {acc*100:.2f}%")
    results[name] = accs

# ── Print all results ────────────────────────────────────────────
print("\n" + "="*60)
print("FULL RESULTS SUMMARY")
print("="*60)
for name, accs in results.items():
    print(f"\n{name}:")
    for s, a in zip(SNRS, accs):
        print(f"  SNR {s:+4d} dB  →  {a*100:.2f}%")

# ── Plot helper ──────────────────────────────────────────────────
def make_plot(ax, names):
    for name in names:
        color, marker = STYLE[name]
        accs = results[name]
        ax.plot(SNRS, [a*100 for a in accs],
                marker=marker, color=color, linewidth=1.8,
                markersize=5 if marker not in ('+','x') else 7,
                label=name)
    ax.set_xlabel("SNR(dB)", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_xticks(SNRS)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}%"))
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)

# ── Fig 2(a): 2D models + AutoSMC* + AutoSMC ───────────────────
fig2a_names = ["MobileNet","ResNet50","DenseNet169","AutoSMC*","AutoSMC"]
fig, ax = plt.subplots(figsize=(7, 5))
make_plot(ax, fig2a_names)
ax.set_title("Fig 2(a) — RADIOML 2016.10A", fontsize=11)
plt.tight_layout()
plt.savefig("fig2a_autosmc_radioml2016_10a.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved → fig2a_autosmc_radioml2016_10a.png")

# ── Fig 2(b): 1D SMC models + AutoSMC* + AutoSMC ───────────────
fig2b_names = ["CNN_Accu_polar","MCLDNN","LSTM","VTCNN2","RN",
               "DAE","STN-ResNeXt","MsmcNet","AutoSMC*","AutoSMC"]
fig, ax = plt.subplots(figsize=(7, 5))
make_plot(ax, fig2b_names)
ax.set_title("Fig 2(b) — RADIOML 2016.10A", fontsize=11)
plt.tight_layout()
plt.savefig("fig2b_autosmc_radioml2016_10a.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved → fig2b_autosmc_radioml2016_10a.png")

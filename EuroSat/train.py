import os
import sys
import csv
import json
import time
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--bands', choices=['rgb', 'ms'], required=True)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--epochs', type=int, default=80)
parser.add_argument('--subset', type=float, default=1.0)
parser.add_argument('--cache', default='cache')
args = parser.parse_args()

CACHE = args.cache
OUT = 'results' if args.subset == 1.0 else 'results_smoke'
BATCH = 128
PATIENCE = 10
LR = 1e-3
FILTERS = (16, 32, 64)

# Indexes for band positions in EuroSAT data (omit 0, 8, 9)
BAND_IDX = {
    'rgb': [3, 2, 1],
    'ms': [1, 2, 3, 4, 5, 6, 7, 10, 11, 12],
}
band_idx = BAND_IDX[args.bands]
n_channels = len(band_idx)

run_name = f'{args.bands}_seed{args.seed}'
run_dir = os.path.join(OUT, run_name)
if os.path.exists(os.path.join(run_dir, 'metrics.json')):
    print(f'{run_name} already complete, skipping run.')
    sys.exit(0)
os.makedirs(run_dir, exist_ok=True)

print(f'\n=== {run_name} | {n_channels} channels | subset {args.subset} ===\n')

# === Seeds ===
keras.utils.set_random_seed(args.seed)
tf.config.experimental.enable_op_determinism()
rng = np.random.default_rng(args.seed)

# === Load cache ===
X = np.load(os.path.join(CACHE, 'X_ms.npy'))
y_all = np.load(os.path.join(CACHE, 'y.npy')).astype('int32')
train_idx = np.load(os.path.join(CACHE, 'train_idx.npy'))
val_idx = np.load(os.path.join(CACHE, 'val_idx.npy'))
test_idx = np.load(os.path.join(CACHE, 'test_idx.npy'))
band_mean = np.load(os.path.join(CACHE, 'band_mean.npy'))[band_idx]
band_std = np.load(os.path.join(CACHE, 'band_std.npy'))[band_idx]
n_classes = int(y_all.max()) + 1

# === Stratified subset for smoke tests ===
if args.subset < 1.0:
    train_idx, _ = train_test_split(train_idx, train_size=args.subset,
                                    stratify=y_all[train_idx], random_state=args.seed)
    val_idx, _ = train_test_split(val_idx, train_size=args.subset,
                                  stratify=y_all[val_idx], random_state=args.seed)
    test_idx, _ = train_test_split(test_idx, train_size=args.subset,
                                   stratify=y_all[test_idx], random_state=args.seed)

# Sorted order
val_ids = np.sort(val_idx)
test_ids = np.sort(test_idx)
print(f'train: {len(train_idx)}, val: {len(val_ids)}, test: {len(test_ids)}')


# === Data pipeline ===
def make_gen(ids, shuffle, augment):
    def gen():
        order = rng.permutation(ids) if shuffle else ids
        for start in range(0, len(order), BATCH):
            b = np.sort(order[start:start + BATCH])
            xb = X[b][..., band_idx].astype('float32')
            xb = (xb - band_mean) / band_std
            if augment:
                m = rng.random(len(b)) < 0.5
                xb[m] = xb[m, :, ::-1]
                m = rng.random(len(b)) < 0.5
                xb[m] = xb[m, ::-1]
                m = rng.random(len(b)) < 0.5
                xb[m] = xb[m].transpose(0, 2, 1, 3)
            yield xb, y_all[b]
    return gen


def make_ds(ids, shuffle, augment):
    sig = (tf.TensorSpec((None, 64, 64, n_channels), tf.float32),
           tf.TensorSpec((None,), tf.int32))
    n_batches = int(np.ceil(len(ids) / BATCH))
    ds = tf.data.Dataset.from_generator(make_gen(ids, shuffle, augment), output_signature=sig)
    ds = ds.apply(tf.data.experimental.assert_cardinality(n_batches))
    return ds.prefetch(2)

train_ds = make_ds(train_idx, shuffle=True, augment=True)
val_ds = make_ds(val_ids, shuffle=False, augment=False)
test_ds = make_ds(test_ids, shuffle=False, augment=False)

# === Model (same for both other than depth of the input) ===
inputs = keras.Input(shape=(64, 64, n_channels))
x = inputs
for f in FILTERS:
    x = layers.Conv2D(f, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(f, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(n_classes, activation='softmax')(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
n_params = model.count_params()
print(f'Parameters: {n_params:,}')

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE,
                                  restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
]

# === Train ===
t0 = time.time()
history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                    callbacks=callbacks, shuffle=False, verbose=2)
train_seconds = time.time() - t0

hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
epochs_run = len(hist['loss'])
best_epoch = int(np.argmin(hist['val_loss'])) + 1

# === Evaluate on test ===

y_prob = model.predict(test_ds, verbose=0)
y_true = y_all[test_ids]
y_pred = y_prob.argmax(axis=1)
test_macro_f1 = f1_score(y_true, y_pred, average='macro')
test_acc = accuracy_score(y_true, y_pred)

print(f'\nTest macro F1-score: {test_macro_f1:.4f} | accuracy: {test_acc:.4f} | '
      f'epochs: {epochs_run} | best epoch: {best_epoch} | training time: {train_seconds / 60:.1f} minutes')

# === Save outputs ===
np.save(os.path.join(run_dir, 'y_prob.npy'), y_prob.astype('float32'))
np.save(os.path.join(run_dir, 'y_true.npy'), y_true)
np.save(os.path.join(run_dir, 'test_ids.npy'), test_ids)
with open(os.path.join(run_dir, 'history.json'), 'w') as f:
    json.dump(hist, f, indent=2)
model.save(os.path.join(run_dir, 'model.keras'))

row = {
    'run': run_name, 'bands': args.bands, 'seed': args.seed, 'subset': args.subset,
    'n_channels': n_channels, 'params': n_params, 'epochs_run': epochs_run,
    'best_epoch': best_epoch, 'train_seconds': round(train_seconds, 1),
    'test_macro_f1': round(test_macro_f1, 6), 'test_acc': round(test_acc, 6)
}

csv_path = os.path.join(OUT, 'results.csv')
write_header = not os.path.exists(csv_path)
with open(csv_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(row))
    if write_header:
        writer.writeheader()
    writer.writerow(row)

with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
    json.dump(row, f, indent=2)

print('Training done.')
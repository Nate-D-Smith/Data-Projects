import os
import json
import numpy as np
import tifffile
from sklearn.model_selection import train_test_split

SRC = 'MS/MS'
OUT = 'cache'
SEED = 42
os.makedirs(OUT, exist_ok=True)

print('\n=== Inventory ===\n')
classes = sorted(
    d for d in os.listdir(SRC) 
    if os.path.isdir(os.path.join(SRC, d))
)
class_to_label = {
    c: i for i, c in enumerate(classes)
}

paths = []
labels = []
for c in classes:
    files = sorted(f for f in os.listdir(
        os.path.join(SRC, c)) if f.endswith('.tif'))
    for f in files:
        paths.append(os.path.join(SRC, c, f))
        labels.append(class_to_label[c])

n = len(paths)
print(f'{n} images found across {len(classes)} classes.\n')
for c in classes:
    print(f'{c}: {labels.count(class_to_label[c])}')

print('\n=== Fill arrays ===\n')

X = np.empty((n, 64, 64, 13), dtype='uint16')
y = np.array(labels, dtype='uint8')

for i, p in enumerate(paths):
    img = tifffile.imread(p)
    assert img.shape == (64, 64, 13), f'Unexpected shape {img.shape} in {p}'
    assert img.dtype == np.uint16, f'Unexpected data type {img.dtype} in {p}'
    X[i] = img
    if (i + 1) % 2000 == 0:
        print(f'--- Read {i + 1} / {n} files ---')

print(f'X shape {X.shape}, dtype {X.dtype}, {X.nbytes / 1e9:.2f} GB')

np.save(os.path.join(OUT, 'X_ms.npy'), X)
np.save(os.path.join(OUT, 'y.npy'), y)
with open(os.path.join(OUT, 'classes.json'), 'w') as f:
    json.dump(classes, f, indent=2)

print('\n=== Stratified 70 / 15 / 15 split ===\n')
idx = np.arange(n)
train_idx, temp_idx = train_test_split(
    idx, test_size=0.3, stratify=y, random_state=SEED)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, stratify=y[temp_idx], random_state=SEED)

np.save(os.path.join(OUT, 'train_idx.npy'), train_idx)
np.save(os.path.join(OUT, 'val_idx.npy'), val_idx)
np.save(os.path.join(OUT, 'test_idx.npy'), test_idx)

print(f'train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)}')
assert len(set(train_idx) & set(val_idx)) == 0
assert len(set(train_idx) & set(test_idx)) == 0
assert len(set(val_idx) & set(test_idx)) == 0

print('\n=== Training split per-band statistics ===\n')
CHUNK = 2000
sorted_train = np.sort(train_idx)
total = np.zeros(13, dtype='float64')
total_sq = np.zeros(13, dtype='float64')

for start in range(0, len(sorted_train), CHUNK):
    chunk = X[sorted_train[start:start + CHUNK]].astype('float64')
    total += chunk.sum(axis=(0, 1, 2))
    total_sq += (chunk ** 2).sum(axis=(0, 1, 2))

count = len(sorted_train) * 64 * 64
band_mean = total / count
band_std = np.sqrt(total_sq / count - band_mean ** 2)

np.save(os.path.join(OUT, 'band_mean.npy'), band_mean.astype('float32'))
np.save(os.path.join(OUT, 'band_std.npy'), band_std.astype('float32'))

names = ['coastal', 'blue', 'green', 'red', 'veg1', 'veg2', 'veg3',
         'nir', 'water_vapor', 'cirrus', 'swir1', 'swir2', 'narrow_nir']

for nm, m, s in zip(names, band_mean, band_std):
    print(f'  {nm:12s} mean {m:8.1f}  std {s:8.1f}')

print('Done.')
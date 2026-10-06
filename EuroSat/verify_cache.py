import os
import json
import numpy as np
import matplotlib.pyplot as plt
 
OUT = 'cache'
 
X = np.load(os.path.join(OUT, 'X_ms.npy'), mmap_mode='r')
y = np.load(os.path.join(OUT, 'y.npy'))
train_idx = np.load(os.path.join(OUT, 'train_idx.npy'))
val_idx = np.load(os.path.join(OUT, 'val_idx.npy'))
test_idx = np.load(os.path.join(OUT, 'test_idx.npy'))
band_mean = np.load(os.path.join(OUT, 'band_mean.npy'))
band_std = np.load(os.path.join(OUT, 'band_std.npy'))
with open(os.path.join(OUT, 'classes.json')) as f:
    classes = json.load(f)
 
print('\n=== Shapes and dtypes ===\n')
print(f'X {X.shape} {X.dtype}, y {y.shape} {y.dtype}')
assert X.shape == (27000, 64, 64, 13)
assert X.dtype == np.uint16

print('\n=== Class counts ===\n')
expected = {
    'AnnualCrop': 3000, 'Forest': 3000, 'HerbaceousVegetation': 3000,
    'Highway': 2500, 'Industrial': 2500, 'Pasture': 2000,
    'PermanentCrop': 2500, 'Residential': 3000, 'River': 2500,
    'SeaLake': 3000,
}
counts = np.bincount(y, minlength=len(classes))
for c, k in zip(classes, counts):
    flag = 'OK' if expected[c] == k else f'MISMATCH (expected {expected[c]})'
    print(f'{c:22s} {k:5d}  {flag}')
    assert expected[c] == k
 
print('\n=== Split proportions ===\n')
full = counts / counts.sum()
for name, ix in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
    part = np.bincount(y[ix], minlength=len(classes)) / len(ix)
    print(f'{name:5s} n={len(ix):5d}  max deviation from full: {np.abs(part - full).max():.4f}')
assert len(train_idx) + len(val_idx) + len(test_idx) == len(y)
 
print('\n=== Band statistics ===\n')
print(f'min std {band_std.min():.1f}')
assert band_std.min() > 1.0, 'A band has near-zero variance'
assert band_mean[9] < 100, 'Index 9 should be B10 cirrus (near zero)'
assert abs(band_mean[12] - band_mean[7]) / band_mean[7] < 0.3, 'Index 12 should be B8A, close to B08'
 
print('\n=== Band order check (NIR vs red) ===\n')
rng = np.random.default_rng(0)
for cls, nir_should_exceed in [('Forest', True), ('SeaLake', False)]:
    rows = np.where(y == classes.index(cls))[0]
    rows = np.sort(rng.choice(rows, 500, replace=False))
    sample = X[rows].astype('float32')
    red = sample[..., 3].mean()
    nir = sample[..., 7].mean()
    print(f'{cls:8s} red {red:7.1f}  nir {nir:7.1f}  ratio {nir / red:.2f}')
    assert (nir > red) == nir_should_exceed, f'Band order looks wrong for {cls}'
 
print('\n=== RGB round-trip ===\n')
k = int(np.where(y == classes.index('Forest'))[0][0])
rgb = X[k][..., [3, 2, 1]].astype('float32')
rgb = np.clip(rgb / np.percentile(rgb, 98), 0, 1)
plt.imshow(rgb)
plt.title(f'Cache index {k} ({classes[y[k]]}), RGB')
plt.axis('off')
plt.show()
 
print('All checks passed.')
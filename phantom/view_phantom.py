import numpy as np
import matplotlib.pyplot as plt

d = np.load("phantom/phantom_raw.npz")
x, y = d["x"], d["y"]
xs, ys = np.unique(x), np.unique(y)
print("Image size:", len(ys), "x", len(xs))

def to_image(v):
    img = np.zeros((len(ys), len(xs)))
    img[np.searchsorted(ys, y), np.searchsorted(xs, x)] = v
    return img

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for a, name in zip(
    ax, ["PD", "T1", "T2"]):
    a.imshow(to_image(d[name]), cmap="gray")
    a.set_title(name)
plt.show()
import numpy as np


def add_noise(kspace, snr, seed=0, ref_mask=None):
    """Add complex Gaussian noise to k-space so the image SNR equals `snr`.

    kspace   : complex array (Ny, Nx)  [if you pass (Nt, Ny, Nx), each frame is treated alike]
    snr      : wanted SNR = mean signal in the reference region / noise sigma
    seed     : fixed random seed so results can be repeated
    ref_mask : True where the signal is measured (default: bright pixels)
    """
    rng = np.random.default_rng(seed)
    # image of the clean data (ortho FFT keeps the noise level the same in k-space and image)
    img = np.fft.ifft2(kspace, axes=(-2, -1), norm="ortho")
    mag = np.abs(img)
    if mag.ndim == 3:
        mag0 = mag[0]
    else:
        mag0 = mag
    if ref_mask is None:
        ref_mask = mag0 > 0.5 * mag0.max()
    signal = mag0[ref_mask].mean()
    sigma = signal / snr                      # std of the real part and of the imaginary part
    noise = sigma * (rng.standard_normal(kspace.shape) + 1j * rng.standard_normal(kspace.shape))
    return kspace + noise


if __name__ == "__main__":
    # self-test: does add_noise really give the SNR we asked for?
    d = np.load("phantom/phantom.npz")
    clean_img = d["PD"] * d["valid"]                       # a fake MR image
    k = np.fft.fft2(clean_img, norm="ortho")               # image -> k-space
    lab = d["label"]
    wm = lab == np.bincount(lab[d["valid"]]).argmax()      # white-matter pixels

    for want in (10, 25, 50, 100):
        noisy_k = add_noise(k, want, seed=1, ref_mask=wm)
        noisy_img = np.fft.ifft2(noisy_k, norm="ortho")
        measured = np.abs(clean_img)[wm].mean() / np.std((noisy_img - clean_img).real)
        print(f"wanted SNR {want:>3}   measured SNR {measured:6.1f}")
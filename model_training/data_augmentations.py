import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter1d


import torch

def torch_interp1d(x, xp, fp):
    """
    Linear interpolation in torch.
    x: query points (...,)
    xp: x-coordinates of data points (T,)
    fp: values at xp (T,)
    Returns: interpolated values (...,)
    """
    # Ensure 1D
    xp = xp.flatten()
    fp = fp.flatten()
    x = x.flatten()

    # clip to domain
    x = torch.clamp(x, xp[0], xp[-1])

    # find right indices
    idx = torch.searchsorted(xp, x) - 1
    idx = torch.clamp(idx, 0, len(xp)-2)

    x0, x1 = xp[idx], xp[idx+1]
    y0, y1 = fp[idx], fp[idx+1]

    slope = (y1 - y0) / (x1 - x0)
    y = y0 + slope * (x - x0)
    return y.reshape_as(x)


def torch_cumtrapz(y, x):
    """
    Torch version of scipy.integrate.cumulative_trapezoid with initial=0.0
    y: (T,)
    x: (T,)
    """
    dx = x[1:] - x[:-1]
    avg = (y[1:] + y[:-1]) * 0.5
    area = avg * dx
    cumsum = torch.cumsum(area, dim=0)
    return torch.cat([torch.zeros(1, device=y.device, dtype=y.dtype), cumsum])


def _time_warp_from_speed_1d_torch(x, t, speed_fn):
    """
    Torch-only version of time-warp.
    x: (T,)
    t: (T,)
    speed_fn: callable(tt)->(T,) returns s(t)
    Returns: x_warp, t_orig, s, tau
    """
    s = speed_fn(t)
    if s.shape != t.shape:
        raise ValueError("El perfil de velocidad debe tener la misma forma que t.")
    if torch.min(s) <= 0:
        raise ValueError("s(t) debe ser > 0 en todo el dominio.")

    tau = torch_cumtrapz(s, t)
    T_final = t[-1] - t[0]
    tau_end = tau[-1]
    if tau_end <= 0:
        raise RuntimeError("τ(T) no válido; revisa s(t).")
    tau = tau * (T_final / tau_end)

    # maps
    t_of_tau = lambda tauq: torch_interp1d(tauq, tau, t)
    x_of_t   = lambda tq: torch_interp1d(tq, t, x)

    t_query = torch.clamp(t, tau[0], tau[-1])
    t_orig  = t_of_tau(t_query)
    x_warp  = x_of_t(t_orig)
    return x_warp, t_orig, s, tau


def augment_time_warp_cosine_torch(
    neural_features,   # torch.Tensor (B,C,T) o (C,T)
    dt,
    alpha=0.6,
    k=None,
    k_range=(1, 5),
    per_channel_same_warp=True
):
    """
    Aplica time-warp coseno s(t)=1+alpha*cos(2π k t / Tdur) conservando longitud.
    Todo implementado en torch.
    """
    if not isinstance(neural_features, torch.Tensor):
        raise ValueError("Input debe ser torch.Tensor")

    orig_device = neural_features.device
    orig_dtype  = neural_features.dtype


    X = neural_features.to(torch.float16)

    # normalizar a (B,C,T)
    if X.ndim == 2:   # (C,T)
        X = X.unsqueeze(0)
        single = True
    elif X.ndim == 3:
        single = False
    else:
        raise ValueError("neural_features debe ser (C,T) o (B,C,T).")

    B, C, T = X.shape
    Tdur = float(dt) * float(T - 1)
    t = torch.linspace(0.0, Tdur, T, dtype=torch.float16, device=X.device)

    warped_out = torch.empty_like(X, dtype=torch.float16)

    for b in range(B):
        k_val = float(k) if k is not None else float(torch.empty(1).uniform_(*k_range).item())

        def cosine_speed(tt):
            return 1.0 + alpha * torch.sin(2.0 * torch.pi * k_val * tt / Tdur)

        if per_channel_same_warp:
            x0 = X[b, 0]
            _, t_orig, s, tau = _time_warp_from_speed_1d_torch(x0, t, cosine_speed)
            for c in range(C):
                warped_out[b, c] = torch_interp1d(t_orig, t, X[b, c])
        else:
            s = cosine_speed(t)
            tau = torch_cumtrapz(s, t)
            tau = tau * (Tdur / tau[-1])
            t_orig = None
            for c in range(C):
                warped_out[b, c], _, _, _ = _time_warp_from_speed_1d_torch(X[b, c], t, lambda _: s)



    warped_out = warped_out.to(device=orig_device, dtype=orig_dtype)
    if single:
        return warped_out[0]
    return warped_out



def gauss_smooth(inputs, device, smooth_kernel_std=2, smooth_kernel_size=100,  padding='same', augmentation =None):
    """
    Applies a 1D Gaussian smoothing operation with PyTorch to smooth the data along the time axis.
    Args:
        inputs (tensor : B x T x N): A 3D tensor with batch size B, time steps T, and number of features N.
                                     Assumed to already be on the correct device (e.g., GPU).
        kernelSD (float): Standard deviation of the Gaussian smoothing kernel.
        padding (str): Padding mode, either 'same' or 'valid'.
        device (str): Device to use for computation (e.g., 'cuda' or 'cpu').
    Returns:
        smoothed (tensor : B x T x N): A smoothed 3D tensor with batch size B, time steps T, and number of features N.
    """
    # Get Gaussian kernel
    inp = np.zeros(smooth_kernel_size, dtype=np.float32)
    inp[smooth_kernel_size // 2] = 1
    gaussKernel = gaussian_filter1d(inp, smooth_kernel_std)
    validIdx = np.argwhere(gaussKernel > 0.01)
    gaussKernel = gaussKernel[validIdx]
    gaussKernel = np.squeeze(gaussKernel / np.sum(gaussKernel))

    # Convert to tensor
    gaussKernel = torch.tensor(gaussKernel, dtype=torch.float32, device=device)
    gaussKernel = gaussKernel.view(1, 1, -1)  # [1, 1, kernel_size]

    # Prepare convolution
    B, T, C = inputs.shape
    inputs = inputs.permute(0, 2, 1)  # [B, C, T]
    gaussKernel = gaussKernel.repeat(C, 1, 1)  # [C, 1, kernel_size]

    # Perform convolution
    smoothed = F.conv1d(inputs, gaussKernel, padding=padding, groups=C) # [B, C, T]
    
    # Data augmentations
    # Window zeroing
    if augmentation:
        # Time warping
        if augmentation['time_warp']:
            if np.random.rand() < 0.2:
                smoothed = augment_time_warp_cosine_torch(
                    smoothed,
                    dt=0.02,
                    alpha=0.5,         # intensidad del warp (0<|alpha|<1)
                    k=None,            # si None, samplea de k_range
                    k_range=(0.2, 5),    # <-- aquí “la variable variable es k”
                    per_channel_same_warp=augmentation['time_warp_per_channel']
                )

        if augmentation['window_zeroing']:
            if np.random.rand() < 0.2:
                win_len = np.random.randint(10, augmentation['window_zeroing_max_size'])
                start = np.random.randint(0, T - win_len)
                smoothed[:, :, start:start + win_len] = 0

        # Channel scaling
        if augmentation['channel_scaling']:
            if np.random.rand() < 0.2:
                scale = np.random.normal(1.0, augmentation['channel_scaling_std'], size=(B, C, 1))
                scale = torch.tensor(scale, dtype=smoothed.dtype, device=device)
                smoothed = smoothed * scale

        

    return smoothed.permute(0, 2, 1)  # [B, T, C]
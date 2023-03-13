"""
Functions needed for audio manipulation
"""
import numpy as np
from numpy import inf
SECONDS = 3

def nassau_window(n):
    x = np.zeros(n, dtype=np.float32)
    nn = n // 2
    for i in range(nn):
        x[i] = np.sin(
            0.5
            * np.pi
            * np.sin(0.5 * np.pi * (i + 0.5) / nn)
            * np.sin(0.5 * np.pi * (i + 0.5) / nn)
        )
    x[nn:] = np.flipud(x[:nn])
    return x

    # # windowing fft/ifft function
    # OLD
    # def stft(data, fft_size=400, step_size=200, padding=True):
    #     # short time fourier transform
    #     if padding == True:
    #         pad = np.zeros(
    #             fft_size,
    #         )
    #         data = np.concatenate((data, pad), axis=0)
    #     window = nassau_window(fft_size)
    #     win_num = (len(data) - fft_size) // step_size
    #     out = np.ndarray((win_num, fft_size), dtype=data.dtype)
    #     for i in range(win_num):
    #         left = int(i * step_size)
    #         right = int(left + fft_size)
    #         out[i] = data[left:right] * window
    #     F = np.fft.rfft(out, axis=1)
    #     return F

    # def istft(F, fft_size=400, step_size=200, padding=True):
    # inverse short time fourier transform
    # data = np.fft.irfft(F, axis=-1)
    # window = nassau_window(fft_size)
    # number_windows = F.shape[0]
    # T = np.zeros((number_windows * step_size + fft_size))
    # for i in range(number_windows):
    #     head = int(i * step_size)
    #     tail = int(head + fft_size)
    #     T[head:tail] = T[head:tail] + data[i, :] * window
    # if padding == True:
    #     T = T[:24000]
    # return T


import torch


# NEW
def istft(F, fft_size=400, step_size=200, padding=True):
    F = F.T
    result = torch.istft(
        input=torch.from_numpy(F),
        n_fft=fft_size,
        hop_length=step_size,
        center=False,
        window=torch.hamming_window(fft_size),
    ).numpy()

    if padding:
        result = result[:24000*SECONDS]

    return result


def stft(data, fft_size=400, step_size=200, padding=True):
    if padding == True:
        pad = np.zeros(
            200,
        )
        data = np.concatenate((data, pad), axis=0)
    result = torch.stft(
        torch.from_numpy(data),
        fft_size,
        hop_length=step_size,
        center=False,
        window=torch.hamming_window(fft_size),
        return_complex=True,
    )
    return result.T.numpy()


def fast_stft(data, power=False, **kwargs):
    # directly transform the wav to the input
    # power law = A**0.3 , to prevent loud audio from overwhelming soft audio
    if power:
        data = power_law(data)
    return real_imag_expand(stft(data))


def fast_istft(F, power=False, **kwargs):
    # directly transform the frequency domain data to time domain data
    # apply power law
    T = istft(real_imag_shrink(F))
    if power:
        T = power_law(T, (1.0 / 0.6))
    return T



def power_law(data, power=0.6):
    # assume input has negative value
    mask = np.zeros(data.shape)
    mask[data >= 0] = 1
    mask[data < 0] = -1
    data = np.power(np.abs(data), power)
    data = data * mask
    return data


def real_imag_expand(c_data, dim="new"):
    # dim = 'new' or 'same'
    # expand the complex data to 2X data with true real and image number
    if dim == "new":
        D = np.zeros((c_data.shape[0], c_data.shape[1], 2))
        D[:, :, 0] = np.real(c_data)
        D[:, :, 1] = np.imag(c_data)
        return D
    if dim == "same":
        D = np.zeros((c_data.shape[0], c_data.shape[1] * 2))
        D[:, ::2] = np.real(c_data)
        D[:, 1::2] = np.imag(c_data)
        return D


def fast_cRM(Fclean, Fmix, K=10, C=0.1):
    """

    :param Fmix: mixed/noisy stft
    :param Fclean: clean stft
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return crm: compressed crm
    """
    M = generate_cRM(Fmix, Fclean)
    crm = cRM_tanh_compress(M, K, C)
    return crm


def generate_cRM(Y, S):
    """

    :param Y: mixed/noisy stft
    :param S: clean stft
    :return: structed cRM
    """
    M = np.zeros(Y.shape)
    epsilon = 1e-8
    # real part
    M_real = np.multiply(Y[:, :, 0], S[:, :, 0]) + np.multiply(Y[:, :, 1], S[:, :, 1])
    square_real = np.square(Y[:, :, 0]) + np.square(Y[:, :, 1])
    M_real = np.divide(M_real, square_real + epsilon)
    M[:, :, 0] = M_real
    # imaginary part
    M_img = np.multiply(Y[:, :, 0], S[:, :, 1]) - np.multiply(Y[:, :, 1], S[:, :, 0])
    square_img = np.square(Y[:, :, 0]) + np.square(Y[:, :, 1])
    M_img = np.divide(M_img, square_img + epsilon)
    M[:, :, 1] = M_img
    return M


def cRM_tanh_compress(M, K=10, C=0.1):
    """
    Recall that the irm takes on vlaues in the range[0,1],compress
    the cRM with hyperbolic tangent
    :param M: crm (298,257,2)
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return crm: compressed crm
    """

    numerator = 1 - np.exp(-C * M)
    numerator[numerator == inf] = 1
    numerator[numerator == -inf] = -1
    denominator = 1 + np.exp(-C * M)
    denominator[denominator == inf] = 1
    denominator[denominator == -inf] = -1
    crm = K * np.divide(numerator, denominator)

    return crm


def fast_icRM(Y, crm, K=10, C=0.1):
    """
    :param Y: mixed/noised stft
    :param crm: DNN output of compressed crm
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return S: clean stft
    """
    M = cRM_tanh_recover(crm, K, C)
    S = np.zeros(np.shape(M))
    S[:, :, 0] = np.multiply(M[:, :, 0], Y[:, :, 0]) - np.multiply(
        M[:, :, 1], Y[:, :, 1]
    )
    S[:, :, 1] = np.multiply(M[:, :, 0], Y[:, :, 1]) + np.multiply(
        M[:, :, 1], Y[:, :, 0]
    )
    return S


def cRM_tanh_recover(O, K=10, C=0.1):
    """

    :param O: predicted compressed crm
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return M : uncompressed crm
    """

    numerator = K - O
    denominator = K + O
    M = -np.multiply((1.0 / C), np.log(np.divide(numerator, denominator)))

    return M


def real_imag_shrink(F, dim="new"):
    # dim = 'new' or 'same'
    # shrink the complex data to combine real and imag number
    F_shrink = np.zeros((F.shape[0], F.shape[1]))
    if dim == "new":
        F_shrink = F[:, :, 0] + F[:, :, 1] * 1j
    if dim == "same":
        F_shrink = F[:, ::2] + F[:, 1::2] * 1j
    return F_shrink


def generate_random_ints(seed: str, max_val: int) -> tuple:
    """
    Generate two random integer values between 0 and max_val using the given seed.

    Parameters:
    seed (str): The seed value for the random number generator.
    max_val (int): The maximum value for the generated integers.

    Returns:
    tuple: A tuple containing two random integer values.
    """
    np.random.seed(int(seed, 36))
    random_ints = np.random.randint(0, max_val, size=2)
    return tuple(random_ints)

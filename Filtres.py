import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal

def calcCoeffFIRPB(w, dB):
    p = np.arange(1, 30000)

    bestdiff = 9999999
    bestp = 0
    bestH = 0

    for pval in p:

        h = 1 / pval

        num = np.e**(complex(0,-w*pval/2))*np.sin(w*pval/2)
        den = np.e**(complex(0,-w/2))*np.sin(w/2)
        H = np.abs(num/den) * h
        #print("N = " + str(Nval) + ", H = " + str(H) + ", pour h = " + str(h))

        diff = np.abs((10**(dB/20))-H)

        if diff < bestdiff:
            bestdiff = diff
            bestp = pval
            bestH = H

    return bestp, bestH, 1/bestp

def filtreFIR(audioSample, fc=0, forcedHVal=0, forcedPVal=0, y_dB=False, xFreq=True, normalized=False, verbose=True):

    w_norm = 2 * np.pi / forcedPVal

    if normalized:
        n = np.arange(0, w_norm * forcedPVal, w_norm)
    elif xFreq:
        step = audioSample.Fe / forcedPVal
        n = np.arange(0, step*forcedPVal, step)
    else:
        n = np.arange(0, forcedPVal, 1)

    FIRpb = np.zeros(forcedPVal)
    index = 0
    for nval in n:
        FIRpb[index] = forcedHVal
        index += 1

    if verbose:
        plotFilter(FIRpb, "Filtre RIF Passe-Bas", n, y_dB=y_dB)

    return FIRpb

def filtrePasseBas(audioSample, fc=0, y_dB=False, xFreq=False, normalized=False, verbose=True):
    N = 1024
    Fe = audioSample.Fe
    w_norm = 2 * np.pi / Fe

    #fc = w_norm * fc
    mc = fc * N / Fe
    kc = 2 * mc + 1

    n = np.arange(0, N, 1)
    if normalized:
        n = n * w_norm
    elif xFreq:
        n = n * Fe / N

    FIRpb = np.zeros(N)

    # Calculer valeurs du filtres
    index = 0
    for nval in n:
        if index == 0:
            FIRpb[index] = kc / N
        else:
            FIRpb[index] = np.sin(np.pi * nval * kc / N) / (N * np.sin(np.pi * nval / N))
        index += 1

    if verbose:
       plotFilter(FIRpb, "Filtre Passe Bas (pour coupe bande)", n, y_dB=y_dB)

    return FIRpb

def filtreCoupeBande(signalInput, y_dB=False, xFreq=True, normalized=False, verbose=False):
    N = 1024
    Fe = signalInput.Fe
    w_norm = 2 * np.pi / Fe

    n = np.arange(0, N, 1)
    m = n

    if xFreq:
        m = m * Fe / N
    elif normalized:
        m = m * w_norm

    fc = 960
    f1 = (1040 - 960) / 2
    f0 = 960 + f1

    wc = fc * w_norm
    w1 = f1 * w_norm
    w0 = f0 * w_norm

    d = np.concatenate([[1], np.zeros(N-1)])
    hlp = filtrePasseBas(signalInput, fc=w1, y_dB=y_dB, xFreq=False, normalized=True, verbose=verbose)
    filtreCB = d - 2 * hlp * np.cos(w0 * n)

    if verbose:
        plotFilter(filtreCB, "Filtre Coupe-Bande", m, y_dB=y_dB)

    return filtreCB


def plotFilter(filtre, filtreName, n, y_dB=False):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title(filtreName + " (Temporel)")
    if y_dB:
        plt.plot(n, 20 * np.log10(np.abs(filtre)))
        plt.ylabel("Amplitude (dB)")
    else:
        plt.plot(n, filtre)
        plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    plt.title(filtreName + " (Fréquentiel)")
    if y_dB:
        FFTFIRpb = np.fft.fft(filtre)
        plt.plot(n, 20 * np.log10(np.abs(FFTFIRpb)))
        plt.ylabel("Amplitude (dB)")
    else:
        plt.plot(n, np.abs(np.fft.fft(filtre)))
        plt.ylabel("Amplitude")
    plt.show()


p, H, h = calcCoeffFIRPB(np.pi/1000, -3)

print("N = " + str(p) + ", H = " + str(H) + ", pour h = " + str(h))

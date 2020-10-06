import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal

def calcCoeffFIRPB(w, dB):
    N = np.arange(1, 30000)

    bestdiff = 9999999
    bestN = 0
    bestH = 0

    for Nval in N:

        h = 1 / Nval

        num = np.e**(complex(0,-w*Nval/2))*np.sin(w*Nval/2)
        den = np.e**(complex(0,-w/2))*np.sin(w/2)
        H = np.abs(num/den) * h
        #print("N = " + str(Nval) + ", H = " + str(H) + ", pour h = " + str(h))

        diff = np.abs((10**(dB/20))-H)

        if diff < bestdiff:
            bestdiff = diff
            bestN = Nval
            bestH = H

    return bestN, bestH, 1/bestN


def calcCoeff2(w, dB):
    N = np.arange(1, 10000)

    bestdiff = 9999999
    bestN = 0
    bestH = 0

    for Nval in N:

        h = np.ones(Nval+1) * (1/(Nval+1))
        H = np.fft.fft(h)
        diff = np.abs((10 ** (dB / 20)) - np.abs(H[int(Nval/2000)]))

        if diff < bestdiff:
            bestdiff = diff
            bestN = Nval
            bestH = np.abs(H[int(Nval/2000)])

    return bestN, bestH, 1 / bestN


def filtrePasseBas(audioSample, fc=0, forcedHVal=0, forcedNVal=0, y_dB=False, xFreq=True, normalized=False, verbose=True):

    if forcedNVal != 0:
        w_norm = 2 * np.pi / forcedNVal

        if normalized:
            n = np.arange(0, w_norm * forcedNVal, w_norm)
        elif xFreq:
            step = audioSample.Fe / forcedNVal
            n = np.arange(0, step*forcedNVal, step)
        else:
            n = np.arange(0, forcedNVal, 1)

        FIRpb = np.zeros(forcedNVal)
        index = 0
        for nval in n:
            FIRpb[index] = forcedHVal
            index += 1

    else:
        Fe = audioSample.Fe
        N = 1024
        w_norm = 2 * np.pi / N

        wc = w_norm * fc
        mc = fc * N / Fe
        kc = 2 * mc + 1

        if normalized:
            n = np.arange(0, w_norm * N, w_norm)
        elif xFreq:
            step = Fe / N
            n = np.arange(0, step*N, step)
        else:
            n = np.arange(0, N, 1)

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
        # Réponse du filtre
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.title("Filtre passe-bas")
        if y_dB:
            plt.plot(n, 20*np.log10(np.abs(FIRpb)))
            plt.ylabel("Amplitude (dB)")
        else:
            plt.plot(n, FIRpb)
            plt.ylabel("Amplitude")

        plt.subplot(2, 1, 2)
        plt.title("Filtre passe-bas (freq)")
        if y_dB:
            FFTFIRpb = np.fft.fft(FIRpb)
            plt.plot(n, 20*np.log10(np.abs(FFTFIRpb)))
            plt.ylabel("Amplitude (dB)")
        else:
            plt.plot(n, np.abs(np.fft.fft(FIRpb)))
            plt.ylabel("Amplitude")

    return FIRpb

def filtreCoupeBande(signalInput, xFreq=True, normalized=False, verbose=False):

    N = 1024
    Fe = signalInput.Fe
    w_norm = 2 * np.pi / N

    fc = 960
    mc = fc * Fe / N

    f1 = (1040 - fc) / 2
    f0 = fc + f1

    #w0 = f0 * 2 * np.pi / Fe
    #w1 = f1 * 2 * np.pi / Fe

    n = np.arange(0, N, 1)
    #n = np.arange(0, w_norm * N, w_norm)
    m = n

    if xFreq:
        step = Fe / N
        m = np.arange(0, N*step, step)

    reponseImp = filtrePasseBas(signalInput, fc=f1, forcedHVal=0, forcedNVal=0, y_dB=False, xFreq=xFreq, normalized=True, verbose=verbose)
    d = np.concatenate([[1], np.zeros(N-1)])

    filtreCB = d - 2 * reponseImp * np.cos(f0 * n)

    if verbose:
        plt.figure()
        plt.subplot(2,1,1)
        plt.title("Filtre coupe-bande temporel")
        plt.plot(m, filtreCB)
        plt.subplot(2,1,2)
        plt.title("Filtre coupe-bande frequentiel")
        plt.plot(m, np.abs(np.fft.fft(filtreCB)))

    return filtreCB


N, H, h = calcCoeffFIRPB(np.pi/1000, -3)
N2, H2, h2 = calcCoeff2(np.pi/1000, -3)

print("N = " + str(N) + ", H = " + str(H) + ", pour h = " + str(h))
print("N = " + str(N2) + ", H = " + str(H2) + ", pour h = " + str(h2))
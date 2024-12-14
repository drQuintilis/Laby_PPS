import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, spectrogram


def dft(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    X = np.dot(e, x)
    return X

def butter_lowpass_filter(data, cutoff, fs, order=5):
    # Projektowanie filtru Butterwortha
    nyquist = 0.5 * fs  # Częstotliwość Nyquista
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Filtrowanie sygnału
    y = filtfilt(b, a, data)
    return y


def butter_highpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_bandpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = [i / nyquist for i in cutoff]
    b, a = butter(order, normal_cutoff, btype='band', analog=False)
    y = filtfilt(b, a, data)
    return y


def sinus():
    # Parametry sygnału
    czas_trwania = 5
    cz_prob = 100  # Częstotliwość próbkowania
    cz_sygnal = 7  # Główna częstotliwość sygnału
    amplituda = 3
    x = np.linspace(0, czas_trwania, cz_prob * czas_trwania)
    y = np.sin(x * cz_sygnal * 2 * np.pi) * amplituda

    # Dodanie dodatkowych częstotliwości
    y += 0.5 * np.sin(x * 10 * 2 * np.pi)  # 10 Hz
    y += 0.2 * np.sin(x * 15 * 2 * np.pi)  # 15 Hz (najwyższa)

    # Filtracja Butterwortha (dolnoprzepustowa)
    cutoff = 8  # Częstotliwość odcięcia (Hz)
    order = 4   # Rząd filtru
    y_low = butter_lowpass_filter(y, cutoff, cz_prob, order)

    # Filtracja Butterwortha (górnoprzepustowa)
    cutoff = 14  # Częstotliwość odcięcia (Hz)
    order = 4  # Rząd filtru
    y_high = butter_highpass_filter(y, cutoff, cz_prob, order)

    # Filtracja Butterwortha (pasmoprzepustowa)
    cutoff = [9, 13]  # Częstotliwość odcięcia (Hz)
    order = 4  # Rząd filtru
    y_band = butter_bandpass_filter(y, cutoff, cz_prob, order)

    # FFT
    y_fft = np.fft.fft(y)
    y_filtered_fft_low = np.fft.fft(y_low)
    y_filtered_fft_high = np.fft.fft(y_high)
    y_filtered_fft_band = np.fft.fft(y_band)
    czestotliwosci = np.fft.fftfreq(len(y), d=1/cz_prob)

    # Wizualizacja w jednym oknie
    plt.figure(figsize=(15, 17))

    # Sygnał w dziedzinie czasu - Oryginalny
    plt.subplot(8, 1, 1)
    plt.plot(x, y)
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.title("Sygnał oryginalny")
    plt.legend()
    plt.grid(True)

    # Sygnał w dziedzinie czasu - Po filtracji dolnej
    plt.subplot(8, 1, 2)
    plt.plot(x, y_low, color='g')
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.title("Sygnał po filtracji lowpass (8 Hz)")
    plt.legend()
    plt.grid(True)

    # Sygnał w dziedzinie czasu - Po filtracji gornej
    plt.subplot(8, 1, 3)
    plt.plot(x, y_high, color='r')
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.title("Sygnał po filtracji highpass (14 Hz)")
    plt.legend()
    plt.grid(True)

    # Sygnał w dziedzinie czasu - Po filtracji pasmowej
    plt.subplot(8, 1, 4)
    plt.plot(x, y_band, color='purple')
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.title("Sygnał po filtracji bandpass (8-14 Hz)")
    plt.legend()
    plt.grid(True)

    # FFT - Oryginalny sygnał
    plt.subplot(8, 1, 5)
    plt.plot(czestotliwosci[:len(czestotliwosci)//2], np.abs(y_fft)[:len(czestotliwosci)//2])
    plt.xlabel("Częstotliwość [Hz]")
    plt.ylabel("Amplituda")
    plt.title("Widmo FFT - Oryginalny sygnał")
    plt.legend()
    plt.grid(True)

    # FFT - Po filtracji dolnej
    plt.subplot(8, 1, 6)
    plt.plot(czestotliwosci[:len(czestotliwosci)//2], np.abs(y_filtered_fft_low)[:len(czestotliwosci)//2], color='g')
    plt.xlabel("Częstotliwość [Hz]")
    plt.ylabel("Amplituda")
    plt.title("Widmo FFT - lowpass")
    plt.legend()
    plt.grid(True)

    # FFT - Po filtracji gornej
    plt.subplot(8, 1, 7)
    plt.plot(czestotliwosci[:len(czestotliwosci)//2], np.abs(y_filtered_fft_high)[:len(czestotliwosci)//2], color='r')
    plt.xlabel("Częstotliwość [Hz]")
    plt.ylabel("Amplituda")
    plt.title("Widmo FFT - highpass")
    plt.legend()
    plt.grid(True)

    # FFT - Po filtracji pasmowej
    plt.subplot(8, 1, 8)
    plt.plot(czestotliwosci[:len(czestotliwosci)//2], np.abs(y_filtered_fft_band)[:len(czestotliwosci)//2], color='purple')
    plt.xlabel("Częstotliwość [Hz]")
    plt.ylabel("Amplituda")
    plt.title("Widmo FFT - bandpass")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    # Generowanie sygnału z trzema składowymi częstotliwościowymi
    y = (3 * np.sin(2 * np.pi * 7 * x)) + (0.5 * np.sin(2 * np.pi * 10 * x)) + (0.3 * np.sin(2 * np.pi * 15 * x))

    # Obliczanie spektrogramu
    frequencies, times, Sxx = spectrogram(y, fs=cz_prob, nperseg=256, noverlap=128)
    #
    # Wizualizacja spektrogramu
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
    plt.colorbar(label="Amplituda [dB]")
    plt.xlabel("Czas [s]")
    plt.ylabel("Częstotliwość [Hz]")
    plt.title("Spektrogram sygnału")

    plt.show()

if __name__ == '__main__':
    sinus()

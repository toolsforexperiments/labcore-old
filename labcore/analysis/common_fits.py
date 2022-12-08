import numpy as np

from plottr.analyzer.fitters.fitter_base import Fit


class ExponentialDecay(Fit):
    @staticmethod
    def model(coordinates, A, of, tau) -> np.ndarray:
        """$A * \exp(-x/\tau) + of$"""
        return A * np.exp(-coordinates/tau) + of

    @staticmethod
    def guess(coordinates, data):

        # offset guess: The mean of the last 10 percent of the data
        of = np.mean(data[-data.size//10:])

        # amplitude guess: difference between max and min.
        A = np.abs(np.max(data) - np.min(data))
        if data[0] < data[-1]:
            A *= -1

        # tau guess: pick the point where we reach roughly 1/e
        one_over_e_val = of + A/3.
        one_over_e_idx = np.argmin(np.abs(data-one_over_e_val))
        tau = coordinates[one_over_e_idx]

        return dict(A=A, of=of, tau=tau)


class ExponentiallyDecayingSine(Fit):
    @staticmethod
    def model(coordinates, A, of, f, phi, tau) -> np.ndarray:
        """$A \sin(2*\pi*(f*x + \phi/360)) \exp(-x/\tau) + of$"""
        return A * np.sin(2 * np.pi * (f * coordinates + phi/360)) * np.exp(-coordinates/tau) + of

    @staticmethod
    def guess(coordinates, data):
        """This guess will ignore the first value because since it usually is not relaiable."""

        # offset guess: The mean of the data
        of = np.mean(data)

        # amplitude guess: difference between max and min.
        A = np.abs(np.max(data) - np.min(data)) / 2.
        if data[0] < data[-1]:
            A *= -1

        # f guess: Maximum of the absolute value of the fourier transform.
        fft_data = np.fft.rfft(data)[1:]
        fft_coordinates = np.fft.rfftfreq(data.size, coordinates[1] - coordinates[0])[1:]

        # note to confirm, could there be multiple peaks? I am always taking the first one here.
        f_max_index = np.argmax(fft_data)
        f = fft_coordinates[f_max_index]

        # phi guess
        phi = -np.angle(fft_data[f_max_index], deg=True)

        # tau guess: pick the point where we reach roughly 1/e
        one_over_e_val = of + A/3.
        one_over_e_idx = np.argmin(np.abs(data-one_over_e_val))
        tau = coordinates[one_over_e_idx]

        return dict(A=A, of=of, phi=phi, f=f, tau=tau)

class Cosine(Fit):
    @staticmethod
    def model(coordinates, A, of, f, phi) -> np.ndarray:
        """$A \sin(2*\pi*(f*x + \phi/360)) + of$"""
        return A * np.cos(2 * np.pi * (f * coordinates + phi/360.)) + of

    @staticmethod
    def guess(coordinates, data):
        """This guess will ignore the first value because since it usually is not relaiable."""

        # offset guess: The mean of the data
        of = np.mean(data)

        # amplitude guess: difference between max and min.
        A = np.abs(np.max(data) - np.min(data)) / 2.

        # f guess: Maximum of the absolute value of the fourier transform.
        fft_data = np.fft.rfft(data)[1:]
        fft_coordinates = np.fft.rfftfreq(data.size, coordinates[1] - coordinates[0])[1:]

        # note to confirm, could there be multiple peaks? I am always taking the first one here.
        f_max_index = np.argmax(np.abs(fft_data))
        f = fft_coordinates[f_max_index]

        # phi guess
        phi = -np.angle(fft_data[f_max_index], deg=True)

        guess = dict(A=A, of=of, phi=phi, f=f)

        return guess


class Gaussian(Fit):
    @staticmethod
    def model(coordinates, x0, sigma, A, of):
        """$A * np.exp(-(x-x_0)^2/(2\sigma^2)) + of"""
        return A * np.exp(-(coordinates - x0) ** 2 / (2 * sigma ** 2)) + of

    @staticmethod
    def guess(coordinates, data):
        # TODO: very crude at the moment, not likely to work well with not-so-nice data.
        of = np.mean(data)
        dev = data - of
        i_max = np.argmax(np.abs(dev))
        x0 = coordinates[i_max]
        A = data[i_max] - of
        sigma = np.abs((coordinates[-1] - coordinates[0])) / 20
        return dict(x0=x0, sigma=sigma, A=A, of=of)

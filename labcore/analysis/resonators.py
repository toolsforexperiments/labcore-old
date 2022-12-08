from typing import Dict, Any

import numpy as np
import scipy
from matplotlib import pyplot as plt
from pathlib import Path

from plottr.analyzer.fitters.fitter_base import Fit


class ReflectionResponse(Fit):

    @staticmethod
    def model(coordinates: np.ndarray, A: float, f_0: float, Q_i: float, Q_e: float, phase_offset: float,
              phase_slope: float):
        """
        Reflection response model derived from input-output theory. For detail, see section 12.2.6 in "Quantum
        and Atom Optics" by Daniel Adam Steck

        Parameters
        ----------
        coordinates
            1d numpy array containing the frequencies in range of sweeping
        A
            amplitude correction of the response
        f_0
            resonant frequency
        Q_i
            internal Q (coupling to losses of the cavity)
        Q_e
            external Q (coupling to output pins)
        phase_offset
            the offset of phase curve which can be seen at the start and end point of the phase diagram of reflection
            response
        phase_slope
            the slope of phase curve which can be seen at the start and end point of the phase diagram of reflection
            response

        Returns
        -------
        numpy array
            the ideal response calculated with the equation
        """
        x = coordinates
        s11_ideal = (1j * (1 - x / f_0) + (Q_i - Q_e) / (2 * Q_e * Q_i)) / (
                    1j * (1 - x / f_0) - (Q_i + Q_e) / (2 * Q_e * Q_i))
        correction = A * np.exp(1j * (phase_offset + phase_slope * (x - f_0)))
        return s11_ideal * correction

    @staticmethod
    def guess(coordinates: np.ndarray, data: np.ndarray):
        """ make an initial guess on parameters based on the measured reflection response data and the input-output
        theory

        Parameters
        ----------
        coordinates
            1d numpy array containing the frequencies in range of sweeping
        data
            1d numpy array containing the complex measured reflection response data

        Returns
        -------
        dict
            a dictionary whose values are the guess on A, f_0, Q_i, Q_e, phase_offset, and phase slope and keys
            contain their names
        """

        amp = np.abs(np.concatenate((data[:data.size // 10], data[-data.size // 10:]))).mean()
        dip_loc = np.argmax(np.abs(np.abs(data) - amp))
        guess_f_0 = coordinates[dip_loc]

        data = moving_average(data)
        depth = amp - np.abs(data[dip_loc])
        width_loc = np.argmin(np.abs(amp - np.abs(data) - depth / 2))
        kappa = 2 * np.abs(coordinates[dip_loc] - coordinates[width_loc])
        guess_Q_tot = guess_f_0 / kappa
        # print(guess_Q_tot)

        [slope, _] = np.polyfit(coordinates[:data.size // 10], np.angle(data[:data.size // 10], deg=False), 1)
        phase_offset = np.angle(data[0]) + slope * (coordinates[dip_loc] - coordinates[0])
        correction = amp * np.exp(1j * phase_offset)
        # print(data[dip_loc]/correction)
        guess_Q_e = 2 * guess_Q_tot / (1 - np.abs(data[dip_loc]/correction))
        guess_Q_i = 1 / (1 / guess_Q_tot - 1 / guess_Q_e)

        return dict(
            f_0=guess_f_0,
            A=amp,
            phase_offset=phase_offset,
            phase_slope=slope,
            Q_i=guess_Q_i,
            Q_e=guess_Q_e
        )

    @staticmethod
    def nphoton(P_cold_dBm: float, Q_e: float, Q_i: float, f_0: float):
        """ calculate the number of photons in the resonator

        Parameters
        ----------
        P_cold_dBm
            the power (in unit of dBm) injected into the cold resonator, with cable loss taken into account
        Q_i
            internal Q (coupling to losses of the cavity)
        Q_e
            external Q (coupling to output pins)
        f_0
            resonant frequency

        Returns
        -------
        float
            the number of photons
        """
        P_cold_W = 1e-3 * 10 ** (P_cold_dBm / 10.)
        Q_tot = 1 / (1 / Q_e + 1 / Q_i)
        photon_number = 2. * P_cold_W * Q_tot ** 2 / (np.pi * scipy.constants.h * f_0 ** 2 * Q_e)
        return photon_number


class HangerResponseBruno(Fit):
    """model from: https://arxiv.org/abs/1502.04082 - pages 6/7."""

    @staticmethod
    def model(coordinates: np.ndarray, A: float, f_0: float, Q_i: float, Q_e_mag: float, theta: float, phase_offset: float,
              phase_slope: float, transmission_slope: float):
        """A (1 + alpha * (x - f_0)/f_0) (1 - Q_l/|Q_e| exp(i \theta) / (1 + 2i Q_l (x-f_0)/f_0)) exp(i(\phi_v f_0
        + phi_0))"""

        x = coordinates
        amp_correction = A * (1 + transmission_slope * (x - f_0)/f_0)
        phase_correction = np.exp(1j*(phase_slope * x + phase_offset))

        if Q_e_mag == 0:
            Q_e_mag = 1e-12
        Q_e_complex = Q_e_mag * np.exp(-1j*theta)
        Q_c = 1./((1./Q_e_complex).real)
        Q_l = 1./(1./Q_c + 1./Q_i)
        response = 1 - Q_l / np.abs(Q_e_mag) * np.exp(1j * theta) / (1. + 2j*Q_l*(x-f_0)/f_0)

        return response * amp_correction * phase_correction

    @staticmethod
    def guess(coordinates, data) -> Dict[str, Any]:
        
        amp = np.abs(np.concatenate((data[:data.size // 10], data[-data.size // 10:]))).mean()
        dip_loc = np.argmax(np.abs(np.abs(data) - amp))
        guess_f_0 = coordinates[dip_loc]
        [guess_transmission_slope, _] = np.polyfit(coordinates[:data.size // 10], np.abs(data[:data.size // 10]), 1)
        amp_correction = amp * (1+guess_transmission_slope*(coordinates-guess_f_0)/guess_f_0)

        data = moving_average(data)
        depth = amp - np.abs(data[dip_loc])
        width_loc = np.argmin(np.abs(amp - np.abs(data) - depth / 2))
        kappa = 2 * np.abs(coordinates[dip_loc] - coordinates[width_loc])
        guess_Q_l = guess_f_0 / kappa
        # print(guess_Q_tot)

        [slope, _] = np.polyfit(coordinates[:data.size // 10], np.angle(data[:data.size // 10], deg=False), 1)
        phase_offset = np.angle(data[0], deg=False) - slope * (coordinates[0]-0)
        phase_correction = np.exp(1j*(slope * coordinates + phase_offset))
        correction = amp_correction * phase_correction
        # print(data[dip_loc]/correction)

        guess_theta = 0.5  # there are deterministic ways of finding it but looking at two symmetric points close to f_r in S21 , but it's kinda unnecessary so I just choose a small value and it works so far
        guess_Q_e_mag = np.abs(-guess_Q_l * np.exp(1j*guess_theta) / (data[dip_loc]/correction[dip_loc]-1))
        guess_Q_c = 1 / np.real(1 / (guess_Q_e_mag*np.exp(-1j*guess_theta)))
        guess_Q_i = 1 / (1 / guess_Q_l - 1 / guess_Q_c)
        
        return dict(
            A = amp,
            f_0 = guess_f_0,
            Q_i = guess_Q_i,
            Q_e_mag = guess_Q_e_mag,
            theta = guess_theta,
            phase_offset = phase_offset,
            phase_slope = slope,
            transmission_slope = guess_transmission_slope,
        )


        # return dict(
        #     A = 1,
        #     f_0 = 1,
        #     Q_i = 1e6,
        #     Q_e = 1e6,
        #     theta = 0,
        #     phase_offset=0,
        #     phase_slope=0,
        #     transmission_slope=0,
        # )


    @staticmethod
    def nphoton(P_cold_dBm: float, Q_e: float, Q_i: float, f_0: float, theta: float):
        P_cold_W = 1e-3 * 10 ** (P_cold_dBm / 10.)
        Q_e_complex = Q_e * np.exp(-1j*theta)
        Q_c = 1./((1/Q_e_complex).real)
        Q_l = 1./(1./Q_c + 1./Q_i)
        return 2 / (scipy.constants.hbar * (2*np.pi*f_0)**2) * Q_l**2 / Q_c * P_cold_W


class TransmissionResponse(Fit):

    @staticmethod
    def model(coordinates: np.ndarray, f_0: float, A: float, Q_t: float, Q_e: float, phase_offset: float,
              phase_slope: float):
        """
        Reflection response model derived from input-output theory. For detail, see section 12.2.6 in "Quantum
        and Atom Optics" by Daniel Adam Steck

        Parameters
        ----------
        coordinates
            1d numpy array containing the frequencies in range of sweeping
        f_0
            resonant frequency
        Q_t
            total Q
        Q_e
            geometric mean of the two coupling Qs (coupling to output pins) multiplied with the total attenuation
            of the signal path.
        phase_offset
            the offset of phase curve which can be seen at the start and end point of the phase diagram of reflection
            response
        phase_slope
            the slope of phase curve which can be seen at the start and end point of the phase diagram of reflection
            response

        Returns
        -------
        numpy array
            the ideal response calculated with the equation
        """
        x = coordinates
        # s21_ideal = (k_e1*k_e2)**0.5 / ( 1j*2*np.pi*(f_0-x) - (k_e1+k_e2+k_i)/2 )
        # s21_ideal = Q_e / (1j*(1-x/f_0)*Q_e**2 - (Q_e2+Q_e1+Q_e1*Q_e2/Q_i)/2)
        correction = A * np.exp(1j * (phase_offset + phase_slope * (x - f_0)))
        s21 = correction * (1j * Q_e * (1. - x/f_0) - .5 * Q_e / Q_t)**(-1)
        return s21

    @staticmethod
    def guess(coordinates: np.ndarray, data: np.ndarray) -> Dict[str, Any]:
        """ make an initial guess on parameters based on the measured reflection response data and the input-output
        theory

        Parameters
        ----------
        coordinates
            1d numpy array containing the frequencies in range of sweeping
        data
            1d numpy array containing the complex measured reflection response data

        Returns
        -------
        dict
            a dictionary whose values are the guess on A, f_0, Q_i, Q_e, phase_offset, and phase slope and keys
            contain their names
        """
        data = moving_average(data)

        # Average the first and last 10% of the data to get the base amplitude
        amp = np.abs(np.concatenate((data[:data.size // 10], data[-data.size // 10:]))).mean()

        # Find the resonance frequency from the max point
        dip_loc = np.argmax(np.abs(np.abs(data) - amp))
        guess_f_0 = coordinates[dip_loc]

        # Find the depth to get kappa and from kappa and f_0 estimate Q_t
        depth = amp - np.abs(data[dip_loc])
        width_loc = np.argmin(np.abs(amp - np.abs(data) - depth / 2))
        kappa = np.abs(coordinates[dip_loc] - coordinates[width_loc])
        guess_Q_t = guess_f_0 / kappa

        # Use Q_t estimate and the max value to get an estimate for Q_e'
        guess_Q_e = -2*guess_Q_t/np.abs(depth)

        # Guess the phase offset and slope
        [guess_slope, _] = np.polyfit(coordinates[:data.size // 10], np.angle(data[:data.size // 10], deg=False), 1)
        guess_phase = np.angle(data[0]) + guess_slope * (coordinates[dip_loc] - coordinates[0])

        #print(guess_phase)

        return dict(
            A=amp,
            f_0=guess_f_0,
            Q_t=guess_Q_t,
            Q_e=-250,
            phase_offset=guess_phase,
            phase_slope=guess_slope,
        )

    # @staticmethod
    # def nphoton(P_cold_dBm: float, Q_e1: float, Q_e2: float, Q_i: float, f_0: float):
    #     """ calculate the number of photons in the resonator
    #
    #     Parameters
    #     ----------
    #     P_cold_dBm
    #         the power (in unit of dBm) injected into the cold resonator, with cable loss taken into account
    #     Q_i
    #         internal Q (coupling to losses of the cavity)
    #     Q_e1
    #         input external Q (coupling to input pins)
    #     Q_e2
    #         output external Q (coupling to output pins)
    #     f_0
    #         resonant frequency
    #
    #     Returns
    #     -------
    #     float
    #         the number of photons
    #     """
    #     P_cold_W = 1e-3 * 10 ** (P_cold_dBm / 10.)
    #     Q_tot = 1 / (1 / Q_e1 + 1 / Q_e2 + 1 / Q_i)
    #     photon_number = 2. * P_cold_W * Q_tot ** 2 / (np.pi * scipy.constants.h * f_0 ** 2 * Q_e1)
    #     return photon_number


def moving_average(a):
    n = a.size//200*2+1
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return np.append(np.append(a[:int((n-1)/2-1)], ret[n - 1:] / n), a[int(-(n-1)/2-1):])


def plot_resonator_response(frequency: np.ndarray, figsize: tuple =(6, 3), f_unit: str = 'Hz', **sparams):
    """ plot the magnitude, phase, and polar diagrams of the data, the model with initially guessed parameters,
    and the fitted curve

    Parameters
    ----------
    coordinates
        1d numpy array containing the frequencies in range of sweeping
    figsize
        size of the figure. Default is (6,3)
    f_unit
        the unit of frequency, often in Hz or GHz. Default is Hz
    sparams
        a dictionary whose values are either a few 1d arrays containing the measured data, values of model with
        initially guessed parameters, and values of the fitted curve, or a few dictionaries, each containing one of
        them and the corresponding plotting, and keys are their names.

    Returns
    -------
    matplotlib.figure
        a figure showing the magnitude, phase, and polar diagrams of the data, the model with initially guessed
        parameters, and the fitted curve
    """
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[2,1])
    mag_ax = fig.add_subplot(gs[0,0])
    phase_ax = fig.add_subplot(gs[1,0], sharex=mag_ax)
    circle_ax = fig.add_subplot(gs[:,1], aspect='equal')

    for name, sparam in sparams.items():
        if isinstance(sparam, np.ndarray):
            data = sparam
            sparam = {}
        elif isinstance(sparam, dict):
            data = sparam.pop('data')
        else:
            raise ValueError(f"cannot accept data of type {type(sparam)}")

        mag_ax.plot(frequency, np.abs(data), label=name, **sparam)
        phase_ax.plot(frequency, np.angle(data, deg=False), **sparam)
        circle_ax.plot(data.real, data.imag, **sparam)

    mag_ax.legend(loc='best', fontsize='x-small')
    mag_ax.set_ylabel('Magnitude')

    phase_ax.set_ylabel('Phase (rad)')
    phase_ax.set_xlabel(f'Frequency ({f_unit})')

    circle_ax.set_xlabel('Re')
    circle_ax.set_ylabel('Im')

    return fig


def fit_and_plot_reflection(f_data: np.ndarray, s11_data: np.ndarray, fn=None, **guesses):
    """ convenience function which does the fitting and plotting (and saving to local directory if given the address)
     in a single call

    Parameters
    ----------
    f_data
        1d numpy array containing the frequencies in range of sweeping
    s11_data
        1d numpy array containing the complex measured reflection response data
    guesses
        (optional) manual guesses on fit parameters

    Returns
    -------
    matplotlib.figure
        a figure showing the magnitude, phase, and polar diagrams of the data, the model with initially guessed
        parameters, and the fitted curve
    """
    fit = ReflectionResponse(f_data, s11_data)
    guess_result = fit.run(dry=True, **guesses)
    guess_y = guess_result.eval()

    fit_result = fit.run(**guesses)
    print(fit_result.lmfit_result.fit_report())

    fit_y = fit_result.eval()

    fig = plot_resonator_response(f_data * 1e-9, f_unit='GHz',
                                  data=dict(data=s11_data, lw=0, marker='.'),
                                  guess=dict(data=guess_y, lw=1, dashes=[1, 1]),
                                  fit=dict(data=fit_y))

    if fn is not None:
        with open(Path(fn.parent, 'fit.txt'), 'w') as f:
            f.write(fit_result.lmfit_result.fit_report())
        fig.savefig(Path(fn.parent, 'fit.png'))
        print('fit result and plot saved')

    return fig

def fit_and_plot_resonator_response(f_data: np.ndarray, s11_data: np.ndarray, response_type: str = 'transmission', fn=None, **guesses):
    """ convenience function which does the fitting and plotting (and saving to local directory if given the address)
     in a single call

    Parameters
    ----------
    f_data
        1d numpy array containing the frequencies in range of sweeping
    s11_data
        1d numpy array containing the complex measured reflection response data
    response_type
        name of the response that we want to fit. The default is transmission response
    guesses
        (optional) manual guesses on fit parameters

    Returns
    -------
    matplotlib.figure
        a figure showing the magnitude, phase, and polar diagrams of the data, the model with initially guessed
        parameters, and the fitted curve
    """
    if response_type == 'transmission':
        fit = TransmissionResponse(f_data, s11_data)
    elif response_type == 'hanger':
        fit = HangerResponseBruno(f_data, s11_data)
    else:
        fit = ReflectionResponse(f_data, s11_data)
    guess_result = fit.run(dry=True, **guesses)
    guess_y = guess_result.eval()

    fit_result = fit.run(**guesses)
    print(fit_result.lmfit_result.fit_report())

    fit_y = fit_result.eval()

    fig = plot_resonator_response(f_data * 1e-9, f_unit='GHz',
                                  data=dict(data=s11_data, lw=0, marker='.'),
                                  guess=dict(data=guess_y, lw=1, dashes=[1, 1]),
                                  fit=dict(data=fit_y))
    print()
    print("===========")
    print()
    if response_type == 'transmission':
        print("Kappa total is", round(fit_result.params['f_0'].value / fit_result.params['Q_t'].value * 1e-6, 3), "MHz")
    if response_type == 'hanger':
        Q_e_mag = fit_result.params['Q_e_mag'].value
        theta = fit_result.params['theta'].value
        print("for your convenience: Q_c = 1/Re{1/Q_e} = ", 1 / np.real( 1 / (Q_e_mag*np.exp(-1j*theta)) ))


    if fn is not None:
        with open(Path(fn.parent, 'fit.txt'), 'w') as f:
            f.write(fit_result.lmfit_result.fit_report())
        fig.savefig(Path(fn.parent, 'fit.png'))
        print('fit result and plot saved')

    return fig

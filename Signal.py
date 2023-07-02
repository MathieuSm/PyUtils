#%% #!/usr/bin/env python3
# Initialization

Version = '01'

Description = """
    Provide different functions for signal analysis

    Version Control:
        01 - Original script

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel, University of Bern

    Date: July 2023
    """

#%% Imports
# Modules import

import argparse
import numpy as np
from numba import njit
import scipy.signal as sig
import matplotlib.pyplot as plt

#%% Functions
# Define functions


def __init__(self):
    pass

def FFT(self, Signal, Sampling, Show=False):

    """
    Analyze signal spectrum in frequency domain
    
    :param Signal: the signal to analyze
    :param Sampling: signal sampling interval (in /s or /m)
    :param Show: Plot the frequency spectrum
    """

    SamplingFrequency = 1 / Sampling
    NormalizedSpectrum = np.fft.fft(Signal) / len(Signal)
    Frequencies = np.fft.fftfreq(Signal.shape[-1], SamplingFrequency)

    RealHalfSpectrum = np.abs(NormalizedSpectrum.real[Frequencies >= 0])
    HalfFrequencies = Frequencies[Frequencies >= 0]

    if Show:
        Figure, Axis = plt.subplots(1,1)
        Axis.semilogx(HalfFrequencies, RealHalfSpectrum, color=(1,0,0))
        Axis.set_xlabel('Frequencies [Hz]')
        Axis.set_ylabel('Amplitude [-]')
        plt.show()

    return HalfFrequencies, RealHalfSpectrum

def DesignFilter(self, Frequency, Order=2):

    """
    Design Butterworth filter according to cut-off frequency and order

    :param Frequency: cut-off frequency of the filter
    :param Order: order of the filter
    """
    
    b, a = sig.butter(Order, Frequency, 'low', analog=True)
    w, h = sig.freqs(b, a)

    Figure, Axis = plt.subplots(1,1)
    Axis.semilogx(w, 20 * np.log10(abs(h)))
    Axis.set_xlabel('Frequency [radians / second]')
    Axis.set_ylabel('Amplitude [dB]')
    Axis.grid(which='both', axis='both')
    Axis.axvline(Frequency, color='green') # cutoff frequency
    Axis.set_ylim([-50,5])
    plt.show()

    return

def Filter(self, Signal, Sampling, Frequency, Order=2, Show=False):
    
    """
    Filter signal and look filtering effect

    :param Signal: signal to filter
    :param Sampling: signal sampling interval (in /s or /m)
    :param Frequency: cut-off frequency
    :param Order: filter order
    :param Show: plot results
    """
    
    SOS = sig.butter(Order, Frequency / Sampling, output='sos')
    FilteredSignal = sig.sosfiltfilt(SOS, Signal)

    if Show:
        Figure, Axis = plt.subplots(1,1)
        Axis.plot(Signal, color=(0,0,0))
        Axis.plot(FilteredSignal, color=(1,0,0))
        plt.show()

    return FilteredSignal

def MaxSlope(self, X, Y=[], WindowWidth=1, StepSize=1):

    if len(Y) == 0:
        Y = X.copy()
        X = np.arange(len(X))

    Slope = NumbaMaxSlope(np.array(X), np.array(Y), int(WindowWidth), StepSize)

    return Slope

@njit
def NumbaMaxSlope(Xdata, Ydata, WindowWidth, StepSize):

    Slopes = []
    Iterations = round((len(Xdata) - WindowWidth) / StepSize)
    
    for i in range(Iterations):

        Start = i * StepSize
        Stop = Start + WindowWidth
        XPoints = Xdata[Start:Stop] 
        YPoints = Ydata[Start:Stop] 
        N = len(XPoints)

        X = np.ones((N,2))
        X[:,1] = XPoints
        Y = np.reshape(YPoints, (N,1))

        X1 = np.linalg.inv(np.dot(X.T, X))
        Intercept, Coefficient = np.dot(X1, np.dot(X.T, Y))
        Slopes.append(Coefficient)

    Slope = max(Slopes)[0]

    return Slope


#%% Main
# Main code

def Main():

    return

#%% Execution part
# Execution as main
if __name__ == '__main__':

    # Initiate the parser with a description
    FC = argparse.RawDescriptionHelpFormatter
    Parser = argparse.ArgumentParser(description=Description, formatter_class=FC)

    # Add long and short argument
    SV = Parser.prog + ' version ' + Version
    Parser.add_argument('-V', '--Version', help='Show script version', action='version', version=SV)

    # Read arguments from the command line
    Arguments = Parser.parse_args()

    Main()
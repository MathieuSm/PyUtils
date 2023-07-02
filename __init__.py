#%% #!/usr/bin/env python3
# Initialization

Version = '01'

Description = """
    Initialization file for the Utils folder

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
import pandas as pd
import matplotlib.pyplot as plt

import Plot
import Read
import Time
import Write
import Signal
import Abaqus
import Tensor
import General
import Morphometry
import Registration


#%% Tuning
# Tune diplay settings

DWidth = 320 # display width in number of character
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', DWidth)
pd.set_option('display.width', DWidth)
np.set_printoptions(linewidth=DWidth,suppress=True,formatter={'float_kind':'{:3}'.format})

plt.rc('font', size=12) # increase slightly plot font size for readability


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
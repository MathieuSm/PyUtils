#%% #!/usr/bin/env python3
# Initialization

Version = '01'

Description = """
    Functions used to print time

    Version Control:
        01 - Original script

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel, University of Bern

    Date: July 2023
    """

#%% Imports
# Modules import

import time
import argparse
import numpy as np


#%% Functions
# Define functions

def __init__(self):
    self.Width = 15
    self.Length = 16
    self.Text = 'Process'
    self.Tic = time.time()
    pass

def Set(self, Tic=None):
    
    if Tic == None:
        self.Tic = time.time()
    else:
        self.Tic = Tic

def Print(self, Tic=None,  Toc=None):

    """
    Print elapsed time in seconds to time in HH:MM:SS format
    :param Tic: Actual time at the beginning of the process
    :param Toc: Actual time at the end of the process
    """

    if Tic == None:
        Tic = self.Tic
        
    if Toc == None:
        Toc = time.time()


    Delta = Toc - Tic

    Hours = np.floor(Delta / 60 / 60)
    Minutes = np.floor(Delta / 60) - 60 * Hours
    Seconds = Delta - 60 * Minutes - 60 * 60 * Hours

    print('\nProcess executed in %02i:%02i:%02i (HH:MM:SS)' % (Hours, Minutes, Seconds))

    return

def Update(self, Progress, Text=''):

    Percent = int(round(Progress * 100))
    Np = self.Width * Percent // 100
    Nb = self.Width - Np

    if len(Text) == 0:
        Text = self.Text
    else:
        self.Text = Text

    Ns = self.Length - len(Text)
    if Ns >= 0:
        Text += Ns*' '
    else:
        Text = Text[:self.Length]
    
    Line = '\r' + Text + ' [' + Np*'=' + Nb*' ' + ']' + f' {Percent:.0f}%'
    print(Line, sep='', end='', flush=True)

def Process(self, StartStop:bool, Text=''):

    if len(Text) == 0:
        Text = self.Text
    else:
        self.Text = Text

    if StartStop*1 == 1:
        print('')
        self.Tic = time.time()
        self.Update(0, Text)

    elif StartStop*1 == 0:
        self.Update(1, Text)
        self.Print()


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
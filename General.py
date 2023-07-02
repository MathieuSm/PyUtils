#%% #!/usr/bin/env python3
# Initialization

Version = '01'

Description = """
    General utility functions

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
import sympy as sp
import SimpleITK as sitk
from pathlib import Path


#%% Functions
# Define functions

def SetDirectories(Name):

    CWD = str(Path.cwd())
    Start = CWD.find(Name)
    WD = Path(CWD[:Start], Name)
    Data = WD / '02_Data'
    Scripts = WD / '03_Scripts'
    Results = WD / '04_Results'

    return WD, Data, Scripts, Results

def RotationMatrix(Phi=0.0, Theta=0.0, Psi=0.0, V=np.zeros(3), A=0):

    if (V != 0).any():
        a = np.cos(A) * np.eye(3)
        b = np.sin(A) * np.array([[0, -V[2], V[1]],[V[2], 0, -V[0]],[-V[1], V[0], 0]])
        c = (1-np.cos(A)) * np.outer(V, V)
        R = np.round(a + b + c, 15)

    else:

        # if list of angles, use numpy for speed
        try:
            len(Phi)
            Phi, Theta, Psi = np.array(Phi), np.array(Theta), np.array(Psi)
            Rx = np.array([[np.ones(len(Phi)),  np.zeros(len(Phi)), np.zeros(len(Phi))],
                           [np.zeros(len(Phi)),        np.cos(Phi),       -np.sin(Phi)],
                           [np.zeros(len(Phi)),        np.sin(Phi),        np.cos(Phi)]])

            Ry = np.array([[ np.cos(Theta),      np.zeros(len(Theta)),        np.sin(Theta)],
                           [np.zeros(len(Theta)), np.ones(len(Theta)), np.zeros(len(Theta))],
                           [-np.sin(Theta),      np.zeros(len(Theta)),        np.cos(Theta)]])

            Rz = np.array([[np.cos(Psi),              -np.sin(Psi), np.zeros(len(Psi))],
                           [np.sin(Psi),               np.cos(Psi), np.zeros(len(Psi))],
                           [np.zeros(len(Psi)), np.zeros(len(Psi)),  np.ones(len(Psi))]])

            R = np.einsum('ijl,jkl->lik',Rz, np.einsum('ijl,jkl->ikl',Ry, Rx))

        # if only float angles, use sympy for more accuracy
        except:
            Rx = sp.Matrix([[1,             0,              0],
                            [0, sp.cos(Phi), -sp.sin(Phi)],
                            [0, sp.sin(Phi),  sp.cos(Phi)]])

            Ry = sp.Matrix([[ sp.cos(Theta), 0, sp.sin(Theta)],
                            [0,             1,              0],
                            [-sp.sin(Theta), 0, sp.cos(Theta)]])

            Rz = sp.Matrix([[sp.cos(Psi), -sp.sin(Psi),     0],
                            [sp.sin(Psi),  sp.cos(Psi),     0],
                            [0,             0,              1]])

            R = Rz * Ry * Rx
    
    return np.array(R, dtype='float')

def GetAngles(R):

    # Compute Euler angles from rotation matrix
    # Assuming R = RxRyRz
    # Adapted from https://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-rotation-matrix

    if len(R.shape) == 2:
        # Special case
        if R[0,2] == 1 or R[0,2] == -1:
            E3 = 0 # Set arbitrarily
            dlta = np.arctan2(R[0,1],R[0,2])

            if R[0,2] == -1:
                E2 = np.pi/2;
                E1 = E3 + dlta

            else:
                E2 = -np.pi/2;
                E1 = -E3 + dlta

        else:
            E2 = - np.arcsin(R[0,2])
            E1 = np.arctan2(R[1,2]/np.cos(E2), R[2,2]/np.cos(E2))
            E3 = np.arctan2(R[0,1]/np.cos(E2), R[0,0]/np.cos(E2))
    
    else:
        E1, E2, E3 = np.zeros(len(R)), np.zeros(len(R)), np.zeros(len(R))
        
        M1 = R[:,0,2] == 1
        M2 = R[:,0,2] == -1
        if sum(M1 + M2) > 0:
            dlta = np.arctan2(R[:,0,1],R[:,0,2])

            if sum(M2) > 0:
                E2[M2] = np.pi/2
                E1[M2] = E3 + dlta

            else:
                E2[M1] = -np.pi/2
                E1[M1] = -E3 + dlta

        else:
            E2 = - np.arcsin(R[:,0,2])
            E1 = np.arctan2(R[:,1,2]/np.cos(E2), R[:,2,2]/np.cos(E2))
            E3 = np.arctan2(R[:,0,1]/np.cos(E2), R[:,0,0]/np.cos(E2))

    return np.array([-E1, -E2, -E3]).T

def GetParameterMap(FileName):

    """
    Builds parameter map according to given file
    """

    File = open(FileName, 'r')
    Text = File.read()
    Start = Text.find('(')
    Stop = Text.find(')')

    ParameterMap = {}
    while Start-Stop+1:
        Line = Text[Start+1:Stop]
        Sep = Line.find(' ')
        Name = Line[:Sep]
        Parameter = Line[Sep+1:]

        if Line[Sep+1:].find(' ')+1:
            ParameterMap[Name] = [P for P in Parameter.split()]

        else:
            ParameterMap[Name] = [Parameter]

        Start = Stop + Text[Stop:].find('(')
        Stop = Start + Text[Start:].find(')')

    File.close()

    return ParameterMap

def Resample(Image, Factor=None, Size=[None], Spacing=[None], Order=0):

    Dimension = Image.GetDimension()
    OriginalSpacing = np.array(Image.GetSpacing())
    OriginalSize = np.array(Image.GetSize())
    PhysicalSize = OriginalSize * OriginalSpacing

    Origin = Image.GetOrigin()
    Direction = Image.GetDirection()

    if Factor:
        NewSize = [round(Size/Factor) for Size in Image.GetSize()] 
        NewSpacing = [PSize/(Size-1) for Size,PSize in zip(NewSize, PhysicalSize)]
    
    elif Size[0]:
        NewSize = Size
        NewSpacing = [PSize/Size for Size, PSize in zip(NewSize, PhysicalSize)]
    
    elif Spacing[0]:
        NewSpacing = Spacing
        NewSize = [np.floor(Size/Spacing).astype('int') + 1 for Size,Spacing in zip(PhysicalSize, NewSpacing)]

    NewArray = np.zeros(NewSize[::-1],'int')
    NewImage = sitk.GetImageFromArray(NewArray)
    NewImage.SetOrigin(Origin - OriginalSpacing/2)
    NewImage.SetDirection(Direction)
    NewImage.SetSpacing(NewSpacing)
  
    Transform = sitk.TranslationTransform(Dimension)
    Resampled = sitk.Resample(Image, NewImage, Transform, Order+1)
    
    return Resampled


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
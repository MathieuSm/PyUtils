#%% #!/usr/bin/env python3
# Initialization

Version = '01'

Description = """
    Provide functions to perform registration using Elastix
    https://simpleelastix.github.io/

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
import SimpleITK as sitk

from Time import Process
from General import RotationMatrix

#%% Functions
# Define functions

def __init__(self):
    self.Echo = True

def Register(self, FixedImage, MovingImage, Type, FixedMask=None, MovingMask=None, Path=None, Dictionary={}):

    if self.Echo:
        Text = Type + ' reg.'
        Process(1, Text)
    PM = sitk.GetDefaultParameterMap(Type)

    # Set standard parameters if not specified otherwise
    if 'MaximumNumberOfIterations' not in Dictionary.keys():
        PM['MaximumNumberOfIterations'] = ['2000']

    if 'FixedImagePyramidSchedule' not in Dictionary.keys():
        PM['FixedImagePyramidSchedule'] = ['50', '20', '10']

    if 'MovingImagePyramidSchedule' not in Dictionary.keys():
        PM['MovingImagePyramidSchedule'] = ['50', '20', '10']

    if 'SP_alpha' not in Dictionary.keys():
        PM['SP_alpha'] = ['0.6']

    if 'SP_A' not in Dictionary.keys():
        PM['SP_A'] = ['1000']

    # Set other defined parameters
    for Key in Dictionary.keys():
        PM[Key] = [str(Item) for Item in Dictionary[Key]]


    # Set Elastix and perform registration
    EIF = sitk.ElastixImageFilter()
    EIF.SetParameterMap(PM)
    EIF.SetFixedImage(FixedImage)
    EIF.SetMovingImage(MovingImage)

    if FixedMask:
        FixedMask = sitk.Cast(FixedMask, sitk.sitkUInt8)
        EIF.SetFixedMask(FixedMask)

    if MovingMask:
        MovingMask = sitk.Cast(MovingMask, sitk.sitkUInt8)
        EIF.SetMovingMask(MovingMask)

    if Path:
        EIF.SetOutputDirectory(Path)
        EIF.LogToConsoleOff()
        EIF.LogToFileOn()

    EIF.Execute()

    # Get results
    Result_Image = EIF.GetResultImage()
    TransformParameters = EIF.GetTransformParameterMap()

    # Print elapsed time
    if self.Echo:
        Process(0, Text)

    return Result_Image, TransformParameters
    
def ComputeInverse(self, FixedImage, TPMFileName, MovingImage=None, Path=None):

    """
    Compute inverse of rigid elastix transform. Manual 6.1.6
    """

    if self.Echo:
        Text = 'Inverse reg.'
        Process(1, Text)

    # Set Elastix and perform registration
    EF = sitk.ElastixImageFilter()
    EF.SetFixedImage(FixedImage)
    EF.SetMovingImage(FixedImage)
    EF.SetInitialTransformParameterFileName(TPMFileName)

    EF.SetParameter('HowToCombineTransforms','Compose')
    EF.SetParameter('MaximumNumberOfIteration','2000')
    EF.SetParameter('FixedImagePyramidSchedule', ['50', '20', '10'])
    EF.SetParameter('MovingImagePyramidSchedule', ['50', '20', '10'])
    EF.SetParameter('SP_alpha', '0.6')
    EF.SetParameter('SP_A', '1000')

    if MovingImage:
        EF.SetParameter('Size', '%i %i %i'%MovingImage.GetSize())
        EF.SetParameter('Spacing', '%f %f %f'%MovingImage.GetSpacing())
        EF.SetParameter('Origin', '%f %f %f'%MovingImage.GetOrigin())
    
    if Path:
        EF.SetOutputDirectory(Path)
        EF.LogToConsoleOff()
        EF.LogToFileOn()

    EF.Execute()
    InvertedTransform = EF.GetTransformParameterMap()[0]
    del InvertedTransform['InitialTransformParametersFileName']

    # Print elapsed time
    if self.Echo:
        Process(0, Text)

    return InvertedTransform

def Apply(self, Image,TransformParameterMap,Path=None,Jacobian=None):

    """
    Apply transform parametermap from elastix to an image
    """

    if self.Echo:
        Text = 'Apply Transform'
        Process(1, Text)

    TIF = sitk.TransformixImageFilter()
    TIF.ComputeDeterminantOfSpatialJacobianOff()
    TIF.SetTransformParameterMap(TransformParameterMap)

    if Jacobian:
        TIF.ComputeDeformationFieldOn()
        TIF.ComputeSpatialJacobianOn()

    else:
        TIF.ComputeDeformationFieldOff()
        TIF.ComputeSpatialJacobianOff()


    if Path:
        TIF.SetOutputDirectory(Path)

    TIF.SetMovingImage(Image)
    TIF.Execute()

    ResultImage = TIF.GetResultImage()

    ResultImage.SetOrigin(np.array(TransformParameterMap[0]['Origin'], float))
    ResultImage.SetSpacing(np.array(TransformParameterMap[0]['Spacing'], float))

    # Print elapsed time
    if self.Echo:
        Process(0, Text)

    return ResultImage

def ApplyCustom(self, Image, TransformParameterMap):

    """
    Apply costum parameter map to an image
    """

    if self.Echo:
        Text = 'Custom transform'
        Process(1, Text)

    # Get transform parameters
    D = np.array(TransformParameterMap['FixedImageDimension'], 'int')[0]
    TP = np.array(TransformParameterMap['TransformParameters'],'float')
    COR = np.array(TransformParameterMap['CenterOfRotationPoint'], 'float')

    # Apply rotation
    R = sitk.VersorRigid3DTransform()
    RM = RotationMatrix(Alpha=TP[0], Beta=TP[1], Gamma=TP[2])
    R.SetMatrix([Value for Value in RM.flatten()])
    R.SetCenter(COR)
    Image_R = sitk.Resample(Image, R)

    # Apply translation
    T = sitk.TranslationTransform(int(D), -TP[3:])
    Image_T = sitk.Resample(Image_R, T)

    if self.Echo:
        Process(0, Text)

    return Image_T
        
def ApplyInverse(self, Image, TransformParameterMap):

    """
    Apply inverse rigid transform from transform parameter map
    """

    if self.Echo:
        Text = 'Inverse transform'
        Process(1, Text)

    # Get transform parameters
    D = np.array(TransformParameterMap['FixedImageDimension'], 'int')[0]
    TP = np.array(TransformParameterMap['TransformParameters'],'float')
    COR = np.array(TransformParameterMap['CenterOfRotationPoint'], 'float')

    # Apply inverse translation
    T = sitk.TranslationTransform(int(D), -COR - TP[3:])
    Image_T = sitk.Resample(Image, T)

    # Apply inverse rotation
    RM = RotationMatrix(Alpha=TP[0], Beta=TP[1], Gamma=TP[2])
    RM_Inv = np.linalg.inv(RM)
    R = sitk.VersorRigid3DTransform()
    R.SetMatrix([Value for Value in RM_Inv.flatten()])
    R.SetCenter([0, 0, 0])

    Image_R = sitk.Resample(Image_T, R)

    # Translate again for center of rotation
    T = sitk.TranslationTransform(int(D), COR)
    Image_T = sitk.Resample(Image_R, T)

    if self.Echo:
        Process(0, Text)

    return Image_T


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
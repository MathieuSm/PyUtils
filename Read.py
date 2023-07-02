#%% #!/usr/bin/env python3
# Initialization

Version = '01'

Description = """
    Script allowing to read different image format

    Version Control:
        01 - Original script

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel, University of Bern

    Date: July 2023
    """

#%% Imports
# Modules import

import vtk
import h5py
import struct
import argparse
import numpy as np
import SimpleITK as sitk
from vtk.util.numpy_support import vtk_to_numpy # type: ignore

from Time import Process


#%% Functions
# Define functions

def __init__(self):
    self.Echo = True

def Get_AIM_Ints(self, File):

    """
    Function by Glen L. Niebur, University of Notre Dame (2010)
    reads the integer data of an AIM file to find its header length
    """

    nheaderints = 32
    File.seek(0)
    binints = File.read(nheaderints * 4)
    header_int = struct.unpack("=32i", binints)

    return header_int

def AIM(self, File):

    """
    Reads an AIM file and provides
    the corresponding itk image additional
    data (i.e. spacing, calibration data, 
    and header)

    from Denis hFE pipeline
    """

    if self.Echo:
        Text = 'Read AIM'
        Process(1, Text)

    # read header
    with open(File, 'rb') as f:
        AIM_Ints = self.Get_AIM_Ints(f)
        # check AIM version
        if int(AIM_Ints[5]) == 16:
            if int(AIM_Ints[10]) == 131074:
                Format = "short"
            elif int(AIM_Ints[10]) == 65537:
                Format = "char"
            elif int(AIM_Ints[10]) == 1376257:
                Format = "bin compressed"
                print("     -> format " + Format + " not supported! Exiting!")
                exit(1)
            else:
                Format = "unknown"
                exit(1)
            Header = f.read(AIM_Ints[2])
            Header_Length = len(Header) + 160
            Extents = (0, AIM_Ints[14] - 1, 0, AIM_Ints[15] - 1, 0, AIM_Ints[16] - 1)
        else:
            print("     -> version 030")
            if int(AIM_Ints[17]) == 131074:
                Format = "short"
                print("     -> format " + Format)
            elif int(AIM_Ints[17]) == 65537:
                Format = "char"
                print("     -> format " + Format)
            elif int(AIM_Ints[17]) == 1376257:
                Format = "bin compressed"
                print("     -> format " + Format + " not supported! Exiting!")
                exit(1)
            else:
                Format = "unknown"
                print("     -> format " + Format + "! Exiting!")
                exit(1)
            Header = f.read(AIM_Ints[8])
            Header_Length = len(Header) + 280
            Extents = (0, AIM_Ints[24] - 1, 0, AIM_Ints[26] - 1, 0, AIM_Ints[28] - 1)

    # collect data from header if existing
    # header = re.sub('(?i) +', ' ', header)
    Header = Header.split('\n'.encode())
    Header.pop(0)
    Header.pop(0)
    Header.pop(0)
    Header.pop(0)
    Scaling = None
    Slope = None
    Intercept = None
    IPLPostScanScaling = 1
    for Line in Header:
        if Line.find(b"Orig-ISQ-Dim-p") > -1:
            origdimp = ([int(s) for s in Line.split(b" ") if s.isdigit()])

        if Line.find("Orig-ISQ-Dim-um".encode()) > -1:
            origdimum = ([int(s) for s in Line.split(b" ") if s.isdigit()])

        if Line.find("Orig-GOBJ-Dim-p".encode()) > -1:
            origdimp = ([int(s) for s in Line.split(b" ") if s.isdigit()])

        if Line.find("Orig-GOBJ-Dim-um".encode()) > -1:
            origdimum = ([int(s) for s in Line.split(b" ") if s.isdigit()])

        if Line.find("Scaled by factor".encode()) > -1:
            Scaling = float(Line.split(" ".encode())[-1])
        if Line.find("Density: intercept".encode()) > -1:
            Intercept = float(Line.split(" ".encode())[-1])
        if Line.find("Density: slope".encode()) > -1:
            Slope = float(Line.split(" ".encode())[-1])
        # if el_size scale was applied, the above still takes the original voxel size. This function works
        # only if an isotropic scaling was applied!!!
        if Line.find('downscaled'.encode()) > -1:
            pass
        elif Line.find("scale".encode()) > -1:
            IPLPostScanScaling = float(Line.split(" ".encode())[-1])
    # Spacing is calculated from Original Dimensions. This is wrong, when the images were coarsened and
    # the voxel size is not anymore corresponding to the original scanning resolution!

    try:
        Spacing = IPLPostScanScaling * (
            np.around(np.asarray(origdimum) / np.asarray(origdimp) / 1000, 5)
        )
    except:
        pass

    # read AIM with vtk
    Reader = vtk.vtkImageReader2()
    Reader.SetFileName(File)
    Reader.SetDataByteOrderToLittleEndian()
    Reader.SetFileDimensionality(3)
    Reader.SetDataExtent(Extents)
    Reader.SetHeaderSize(Header_Length)
    if Format == "short":
        Reader.SetDataScalarTypeToShort()
    elif Format == "char":
        Reader.SetDataScalarTypeToChar()
    Reader.SetDataSpacing(Spacing)
    Reader.Update()
    VTK_Image = Reader.GetOutput()


    # Convert VTK to numpy
    Data = VTK_Image.GetPointData().GetScalars()
    Dimension = VTK_Image.GetDimensions()
    Numpy_Image = vtk_to_numpy(Data)
    Numpy_Image = Numpy_Image.reshape(Dimension[2], Dimension[1], Dimension[0])

    # Y symmetry (thanks Michi for notifying this!)
    Numpy_Image = Numpy_Image[:,::-1,:]
    
    # Converty numpy to ITK image
    Image = sitk.GetImageFromArray(Numpy_Image)
    Image.SetSpacing(Spacing)
    Image.SetOrigin([0.0, 0.0, 0.0])

    AdditionalData = {'Scaling':Scaling,
                    'Slope':Slope,
                    'Intercept':Intercept,
                    'Header':Header}

    if self.Echo:
        Process(0, Text)

    return Image, AdditionalData

def ISQ(self, File, BMD=False, Info=False):

    """
    This function read an ISQ file from Scanco and return an ITK image and additional data.
    
    Adapted from https://github.com/mdoube/BoneJ/blob/master/src/org/bonej/io/ISQReader.java
    
    Little endian byte order (the least significant bit occupies the lowest memory position.
    00   char    check[16];              // CTDATA-HEADER_V1
    16   int     data_type;
    20   int     nr_of_bytes;
    24   int     nr_of_blocks;
    28   int     patient_index;          //p.skip(28);
    32   int     scanner_id;				//p.skip(32);
    36   int     creation_date[2];		//P.skip(36);
    44   int     dimx_p;					//p.skip(44);
    48   int     dimy_p;
    52   int     dimz_p;
    56   int     dimx_um;				//p.skip(56);
    60   int     dimy_um;
    64   int     dimz_um;
    68   int     slice_thickness_um;		//p.skip(68);
    72   int     slice_increment_um;		//p.skip(72);
    76   int     slice_1_pos_um;
    80   int     min_data_value;
    84   int     max_data_value;
    88   int     mu_scaling;             //p.skip(88);  /* p(x,y,z)/mu_scaling = value [1/cm]
    92	 int     nr_of_samples;
    96	 int     nr_of_projections;
    100  int     scandist_um;
    104  int     scanner_type;
    108  int     sampletime_us;
    112  int     index_measurement;
    116  int     site;                   //coded value
    120  int     reference_line_um;
    124  int     recon_alg;              //coded value
    128  char    name[40]; 		 		//p.skip(128);
    168  int     energy;        /* V     //p.skip(168);
    172  int     intensity;     /* uA    //p.skip(172);
    ...
    508 int     data_offset;     /* in 512-byte-blocks  //p.skip(508);
    * So the first 16 bytes are a string 'CTDATA-HEADER_V1', used to identify
    * the type of data. The 'int' are all 4-byte integers.
    *
    * dimx_p is the dimension in pixels, dimx_um the dimensions in micrometer
    *
    * So dimx_p is at byte-offset 40, then dimy_p at 44, dimz_p (=number of
    * slices) at 48.
    *
    * The microCT calculates so called 'x-ray linear attenuation' values. These
    * (float) values are scaled with 'mu_scaling' (see header, e.g. 4096) to
    * get to the signed 2-byte integers values that we save in the .isq file.
    *
    * e.g. Pixel value 8192 corresponds to lin. att. coeff. of 2.0 [1/cm]
    * (8192/4096)
    *
    * Following to the headers is the data part. It is in 2-byte short integers
    * (signed) and starts from the top-left pixel of slice 1 to the left, then
    * the next line follows, until the last pixel of the last sclice in the
    * lower right.
    """

    if self.Echo:
        Text = 'Read ISQ'
        Process(1, Text)

    try:
        f = open(File, 'rb')
    except IOError:
        print("\n **ERROR**: ISQReader: intput file ' % s' not found!\n\n" % File)
        print('\n E N D E D  with ERRORS \n\n')

    for Index in np.arange(0, 200, 4):
        f.seek(Index)
        #print('Index %s :          %s' % (Index, struct.unpack('i', f.read(4))[0]))
        f.seek(Index)

    f.seek(32)
    CT_ID = struct.unpack('i', f.read(4))[0]

    if CT_ID != 6020:
        print('!!! unknown muCT -> no Slope and Intercept known !!!')

    f.seek(28)
    #    sample_nb = struct.unpack('i', f.read(4))[0]

    f.seek(108)
    Scanning_time = struct.unpack('i', f.read(4))[0] / 1000

    f.seek(168)
    Energy = struct.unpack('i', f.read(4))[0] / 1000.

    f.seek(172)
    Current = struct.unpack('i', f.read(4))[0]

    f.seek(44)
    X_pixel = struct.unpack('i', f.read(4))[0]

    f.seek(48)
    Y_pixel = struct.unpack('i', f.read(4))[0]

    f.seek(52)
    Z_pixel = struct.unpack('i', f.read(4))[0]

    f.seek(56)
    Res_General_X = struct.unpack('i', f.read(4))[0]
    #print('Resolution general X in mu: ', Res_General_X)

    f.seek(60)
    Res_General_Y = struct.unpack('i', f.read(4))[0]
    #print('Resolution general Y in mu: ', Res_General_Y)

    f.seek(64)
    Res_General_Z = struct.unpack('i', f.read(4))[0]
    #print('Resolution general Z in mu: ', Res_General_Z)

    Res_X = Res_General_X / float(X_pixel)
    Res_Y = Res_General_Y / float(Y_pixel)
    Res_Z = Res_General_Z / float(Z_pixel)

    Header_Txt = ['scanner ID:                 %s' % CT_ID,
                'scaning time in ms:         %s' % Scanning_time,
                'scaning time in ms:         %s' % Scanning_time,
                'Energy in keV:              %s' % Energy,
                'Current in muA:             %s' % Current,
                'nb X pixel:                 %s' % X_pixel,
                'nb Y pixel:                 %s' % Y_pixel,
                'nb Z pixel:                 %s' % Z_pixel,
                'resolution general X in mu: %s' % Res_General_X,
                'resolution general Y in mu: %s' % Res_General_Y,
                'resolution general Z in mu: %s' % Res_General_Z,
                'pixel resolution X in mu:   %.2f' % Res_X,
                'pixel resolution Y in mu:   %.2f' % Res_Y,
                'pixel resolution Z in mu:   %.2f' % Res_Z]
    #    np.savetxt(inFileName.split('.')[0]+'.txt', Header_Txt)

    if Info:
        Write_File = open(File.split('.')[0] + '_info.txt', 'w')
        for Item in Header_Txt:
            Write_File.write("%s\n" % Item)
        Write_File.close()

    f.seek(44)
    Header = np.zeros(6)
    for i in range(0, 6):
        Header[i] = struct.unpack('i', f.read(4))[0]
    #print(Header)

    ElementSpacing = [Header[3] / Header[0] / 1000, Header[4] / Header[1] / 1000, Header[5] / Header[2] / 1000]
    f.seek(508)

    HeaderSize = 512 * (1 + struct.unpack('i', f.read(4))[0])
    f.seek(HeaderSize)


    VoxelModel = np.fromfile(f, dtype='i2')
    # VoxelModel = np.fromfile(f, dtype=np.float)

    NDim = [int(Header[0]), int(Header[1]), int(Header[2])]
    LDim = [float(ElementSpacing[0]), float(ElementSpacing[1]), float(ElementSpacing[2])]

    AdditionalData = {'-LDim': LDim,
                    '-NDim': NDim,
                    'ElementSpacing': LDim,
                    'DimSize': NDim,
                    'HeaderSize': HeaderSize,
                    'TransformMatrix': [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    'CenterOfRotation': [0.0, 0.0, 0.0],
                    'Offset': [0.0, 0.0, 0.0],
                    'AnatomicalOrientation': 'LPS',
                    'ElementType': 'int16',
                    'ElementDataFile': File}

    #print('\nReshape data')
    #Tic = time.time()

    try:
        VoxelModel = VoxelModel.reshape((NDim[2], NDim[1], NDim[0]))
        f.close()
        del f

    except:
        # if the length does not fit the dimensions (len(VoxelModel) != NDim[2] * NDim[1] * NDim[0]),
        # add an offset with seek to reshape the image -> actualise length, delta *2 = seek

        Offset = (len(VoxelModel) - (NDim[2] * NDim[1] * NDim[0]))
        f.seek(0)
        VoxelModel = np.fromfile(f, dtype='i2')

        if self.Echo:
            print('len(VoxelModel) = ', len(VoxelModel))
            print('Should be ', (NDim[2] * NDim[1] * NDim[0]))
            print('Delta:', len(VoxelModel) - (NDim[2] * NDim[1] * NDim[0]))

        f.seek((len(VoxelModel) - (NDim[2] * NDim[1] * NDim[0])) * 2)
        VoxelModel = np.fromfile(f, dtype='i2')
        f.close()
        del f

        VoxelModel = VoxelModel.reshape((NDim[2], NDim[1], NDim[0]))
        # the image is flipped by the Offset --> change the order to obtain the continuous image:
        VoxelModel = np.c_[VoxelModel[:, :, -Offset:], VoxelModel[:, :, :(VoxelModel.shape[2] - Offset)]]


    if self.Echo:
        Process(0,Text)
        print('\nScanner ID:                 ', CT_ID)
        print('Scanning time in ms:        ', Scanning_time)
        print('Energy in keV:              ', Energy)
        print('Current in muA:             ', Current)
        print('Nb X pixel:                 ', X_pixel)
        print('Nb Y pixel:                 ', Y_pixel)
        print('Nb Z pixel:                 ', Z_pixel)
        print('Pixel resolution X in mu:    %.2f' % Res_X)
        print('Pixel resolution Y in mu:    %.2f' % Res_Y)
        print('Pixel resolution Z in mu:    %.2f' % Res_Z)


    if CT_ID == 6020 and BMD is True:
        # BE CAREFULL, THIS IS FOR BMD CONVERSION:
        if self.Echo:
            print('muCT 100 of ISTB detected, IS IT CORRECT?')
        Slope = 369.154  # ! ATTENTION, dependent on voltage, Current and time!!!
        Intercept = -191.56
        try:
            VoxelModel = VoxelModel.astype('i4')
            VoxelModel *= Slope
            VoxelModel += Intercept
        except:
            print('\n********* memory not sufficient for BMD values ************\n')

    # Convert numpy array to image
    Image = sitk.GetImageFromArray(VoxelModel)
    Image.SetSpacing(LDim[::-1])
    Image.SetOrigin([0.0, 0.0, 0.0])

    return Image, AdditionalData

def XDMF(self, File):

    h5f = h5py.File(File[:-5] + '.h5','r')
    Connectivity = h5f['/Connectivity'][:,:]
    Coordinates = h5f['/Coordinates'][:,:]
    TimeSteps = [int(k) for k in h5f['/Displacement'].keys()]
    TimeSteps.sort()
    Shape = (len(TimeSteps), Coordinates.shape[0], Coordinates.shape[1])
    Displacements = np.zeros(Shape)
    for i, Time in enumerate(TimeSteps):
        Displacements[i] = h5f['/Displacement'].get(str(Time))[:,:][:-1]
    ID = np.zeros((Shape[0], Connectivity.shape[0]))
    for i, Time in enumerate(TimeSteps):
        ID[i] = h5f['/ID'].get(str(Time))[:]
    SC = np.zeros((Shape[0], Connectivity.shape[0]))
    for i, Time in enumerate(TimeSteps):
        SC[i] = h5f['/SC'].get(str(Time))[:]

    Centers = np.zeros((Connectivity.shape[0], 3))
    for i, C in enumerate(Connectivity):
        Centers[i] = np.mean(Coordinates[C], axis=0)

    Xc = list(np.unique(Centers[:,0]))
    Yc = list(np.unique(Centers[:,1]))
    Zc = list(np.unique(Centers[:,2]))
    Xc.sort(), Yc.sort(), Zc.sort()

    SphericalCompression = np.zeros((Shape[0], len(Xc), len(Yc), len(Zc)))
    IsovolumicDeformation = np.zeros((Shape[0], len(Xc), len(Yc), len(Zc)))
    for i, C in enumerate(Centers):
        Xi = Xc.index(C[0])
        Yi = Yc.index(C[1])
        Zi = Zc.index(C[2])
        SphericalCompression[:, Xi, Yi, Zi] = SC[:,i]
        IsovolumicDeformation[:, Xi, Yi, Zi] = ID[:,i]

    return Coordinates, Displacements, SphericalCompression, IsovolumicDeformation


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
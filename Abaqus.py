#%% #!/usr/bin/env python3
# Initialization

Version = '01'

Description = """
    Provide functions to perform Abaqus simulations and
    extract results

    Version Control:
        01 - Original script

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel, University of Bern

    Date: July 2023
    """

#%% Imports
# Modules import

import os
import argparse
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path
import SimpleITK as sitk
from multiprocessing import Pool

from Time import Process, Update

#%% Functions
# Define functions

def __init__(self):

        self.NLGEOM = 'YES'
        self.MaxINC = 1000

        self.StepText = """
**
*STEP,AMPLITUDE=RAMP,UNSYMM=YES,INC={MAXINC},NLGEOM={NL}
***********************************************************
**               INCLUDE
***********************************************************
*INCLUDE, input={BCsFile}
***********************************************************
*OUTPUT,FIELD
*ELEMENT OUTPUT, POSITION=CENTROIDAL
SDV2,
SDV15,
SDV16,
SDV17,
SDV18,
SDV22,
SDV23,
SDV24,
SDV25,
SDV26,
SDV27,
SDV28,
SDV29,
SDV30,
SDV31,
S,
LE,
COORD,
*NODE OUTPUT
U,
RF,
CF,
**
*OUTPUT, HISTORY
*NODE OUTPUT, NSET=REF_NODE
U,
RF,
*NODE PRINT, NSET=REF_NODE, FREQUENCY=1, SUMMARY=NO
U,
RF
CF
*END STEP"""

        return

def ReadDAT(self, File):

    """
    Read .dat file from abaqus and extract reference point data
    """

    try:
        with open(File) as F:
            Text = F.read()

            Values = []
            Condition = Text.find('U1') + 1
            Start = Text.find('U1')
            Steps, Increments = [], []
            Step, Increment = 1, 1
            while Condition:

                Inc = int(Text[Start-349:Start-347])
                if Inc < Increment:
                    Step += 1
                Increment = Inc
                Increments.append(Increment)
                Steps.append(Step)

                for i in range(6):
                    iStart = Start + 125 + 15*i
                    iStop = iStart + 14
                    Values.append(float(Text[iStart : iStop]))

                    iStart += Text[Start:].find('RF1')
                    iStop += Text[Start:].find('RF1')
                    Values.append(float(Text[iStart : iStop]))

                Start = iStop + Text[iStop:].find('U1')
                Condition = Text[iStop:].find('U1') + 1

            Values = np.array(Values)
            Cols = 12
            Rows = Values.size // Cols
            Values = np.reshape(Values,(Rows,Cols))

            ColNames = []
            for i in range(3):
                for V in ['U', 'F']:
                    ColNames.append(V + str(i+1))
            for i in range(3):
                for V in ['R', 'M']:
                    ColNames.append(V + str(i+1))

            Data = pd.DataFrame()
            for iName, Name in enumerate(ColNames):
                Data[Name] = np.concatenate([[0], Values[:,iName]])
            
            Data.columns = ['X', 'FX', 'Y', 'FY', 'Z', 'FZ', 'Phi', 'MX', 'Theta', 'MY', 'Psi', 'MZ']

            Data['Step'] = np.concatenate([[0],Steps])
            Data['Increment'] = np.concatenate([[0],Increments])

        return Data

    except FileNotFoundError:
        print('File' + File + 'does not exist')

        return

def WriteRefNodeBCs(self, FileName, DOFs, Values, BCType='DISPLACEMENT', Parameters=[0.05, 1, 5e-05, 0.05]):

    """
    Write boundary conditions for a reference node called REF_NODE
    :param FileName: Name of the BCs file
                        Type: str
    :param DOFs: Degrees of freedom to constrain
                    Type: List of ints
    :param Values: Constrain value for each DOF listed
                    Type: List of floats
    :param BCType: Type of boundary condition
                    Type: str
                    Either 'DISPLACEMENT' (Dirichlet)
                    or 'FORCE' (Neumann)
    :param Parameters: Step parameters
                        P1: Start step size
                        P2: Time for displacement
                        P3: Min step size
                        P4: Max step size
                        Type: List of floats

    """

    with open(FileName, 'w') as F:
        F.write('*STATIC\n')

        F.write(str(Parameters[0])+ ', ')
        F.write(str(Parameters[1])+ ', ')
        F.write(str(Parameters[2])+ ', ')
        F.write(str(Parameters[3])+ '\n')

        F.write('*BOUNDARY, TYPE=' + BCType + '\n')

        for DOF, Value in zip(DOFs, Values):
            Line = str(DOF) + ', ' + str(DOF) + ', ' + str(Value)
            F.write('REF_NODE, ' + Line + '\n')

    return

def AddStep(self, File, Step, DOFs, Values, BCType='DISPLACEMENT', Parameters=[0.05, 1, 5e-05, 0.05]):

    self.WriteRefNodeBCs(Step, DOFs, Values, BCType, Parameters)

    Context = {'MAXINC':self.MaxINC,
                'NL':self.NLGEOM,
                'BCsFile':Step}
    Text = self.StepText.format(**Context)

    with open(File, 'a') as F:
        F.write(Text)

    return

def RemoveSteps(self, File, NSteps=1):

    with open(File, 'r') as F:
        Text = F.read()

    if Text.find('*STEP') == -1:
        print('\nNo step in file!')

    else:
        Start = 0
        Index = Text.find('*STEP', Start)
        Indices = []
        while bool(Index+1):
            Indices.append(Index)
            Start = Index + len('*STEP')
            Index = Text.find('*STEP', Start)

        if NSteps > len(Indices):
            Index = Indices[0]
        else:
            Index = np.array(Indices)[-NSteps]
            
        with open(File, 'w') as F:
            F.write(Text[:Index - len('*STEP') + 1])

    return

def WriteUMAT(self, UMATPath, Name='Elastic'):

    if Name == 'Elastic':
        Text = """C================================================================================================
C
C                          UMAT FOR ISOTROPIC ELASTIC CONSTITUTIVE LAW
C
C     FOUND ON https://simplifiedfem.wordpress.com/about/tutorial-write-a-simple-umat-in-abaqus/
C
C                  ADAPTED BY MATHIEU SIMON - ISTB - UNIVERSITY OF BERN - 2021
C
C================================================================================================
C
C
SUBROUTINE UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD,
1 RPL,DDSDDT,DRPLDE,DRPLDT,
2 STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,CMNAME,
3 NDI,NSHR,NTENS,NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT,
4 CELENT,DFGRD0,DFGRD1,NOEL,NPT,LAYER,KSPT,JSTEP,KINC)
C
C
INCLUDE 'ABA_PARAM.INC'
C
C
CHARACTER*80 CMNAME
DIMENSION DDSDDE(NTENS,NTENS),DDSDDT(NTENS),DRPLDE(NTENS),
1 STRAN(NTENS),DSTRAN(NTENS),TIME(2),PREDEF(1),DPRED(1),
2 PROPS(NPROPS),COORDS(3),DROT(3,3), JSTEP(4)
C     
C     
INTEGER NSTATV, KSTEP, KINC
DOUBLE PRECISION STRESS(NTENS),STATEV(NSTATV)
DOUBLE PRECISION DFGRD0(3,3),DFGRD1(3,3)
C
C
C     ELASTIC USER SUBROUTINE
PARAMETER (ONE=1.0D0, TWO=2.0D0)
C
C
C    Get engineering variables
    E=PROPS(1)
    ANU=PROPS(2)
C
C
C    Compute lamé parameters
    ALAMBDA=E/(ONE+ANU)/(ONE-TWO*ANU)
    BLAMBDA=(ONE-ANU)
    CLAMBDA=(ONE-TWO*ANU)
C
C
C    Initialize material stiffness matrix
        DO I=1,NTENS
        DO J=1,NTENS
        DDSDDE(I,J)=0.0D0
        ENDDO
        ENDDO
C
C
C    Update stiffness matrx
            DDSDDE(1,1)=(ALAMBDA*BLAMBDA)
            DDSDDE(2,2)=(ALAMBDA*BLAMBDA)
            DDSDDE(3,3)=(ALAMBDA*BLAMBDA)
            DDSDDE(4,4)=(ALAMBDA*CLAMBDA)
            DDSDDE(5,5)=(ALAMBDA*CLAMBDA)
            DDSDDE(6,6)=(ALAMBDA*CLAMBDA)
            DDSDDE(1,2)=(ALAMBDA*ANU)
            DDSDDE(1,3)=(ALAMBDA*ANU)
            DDSDDE(2,3)=(ALAMBDA*ANU)
            DDSDDE(2,1)=(ALAMBDA*ANU)
            DDSDDE(3,1)=(ALAMBDA*ANU)
            DDSDDE(3,2)=(ALAMBDA*ANU)
C
C
C     Update stress tensor
    DO I=1,NTENS
        DO J=1,NTENS
        STRESS(I)=STRESS(I)+DDSDDE(I,J)*DSTRAN(J)
        ENDDO
    ENDDO
C
C        
C     Deformation gradient
STATEV(1) = DFGRD1(1,1)
STATEV(2) = DFGRD1(1,2)
STATEV(3) = DFGRD1(1,3)
STATEV(4) = DFGRD1(2,1)
STATEV(5) = DFGRD1(2,2)
STATEV(6) = DFGRD1(2,3)
STATEV(7) = DFGRD1(3,1)
STATEV(8) = DFGRD1(3,2)
STATEV(9) = DFGRD1(3,3)
C
C
    RETURN
    END"""

    with open(str(Path(UMATPath, (Name + '.f'))), 'w') as File:
        File.write(Text)

def uFE(self, BinImage, FileName, UMAT='Elastic', Cap=False):

    Text = 'Create uFE file'
    Process(1,Text)

    # Define directory and parameters
    Compression = 0.001 # Relative compression
    E = 1E4             # Young's modulus
    Nu = 0.3            # Poisson's ratio

    # Define input file name and create it
    if os.name == 'nt':
        Last = FileName[::-1].find('\\')
        NProc = 4
    else:
        Last = FileName[::-1].find('/')
        NProc = 12
    UMATPath = FileName[:-Last]
    self.WriteUMAT(UMATPath, UMAT)

    # Compute nodes coordinates
    BinArray = sitk.GetArrayFromImage(BinImage)
    Spacing = np.round(BinImage.GetSpacing(),3)

    if len(np.unique(BinArray)) > 2:
        print('Image is not binary')
        return
    else:
        BinArray = BinArray.astype(int)
    
    # Add bottom and top cap
    if Cap:
        CapSize = 5
        BinArray = np.pad(BinArray, ((CapSize,CapSize), (0,0), (0,0)), constant_values=(125))
        BinArray = BinArray.astype('uint8')

    # Perform elements and nodes mapping
    Update(1/6,'Perform mapping')
    Nodes, Coords, Elements, ElementsNodes = NumbaMapping(BinArray)
    NodesNeeded = np.unique(ElementsNodes[BinArray.astype(bool)])

    ElementsNodes = ElementsNodes[BinArray.astype(bool)]
    Elements = Elements[BinArray.astype(bool)]
    Materials = BinArray[BinArray.astype(bool)]
    Coords = Coords[np.isin(Nodes,NodesNeeded)]
    Nodes = Nodes[np.isin(Nodes,NodesNeeded)]

    # Sort nodes according to coordinates
    Indices = np.lexsort((-Coords[:,1],-Coords[:,2],-Coords[:,0]))
    Coords = np.round(Coords[Indices] * Spacing,3)
    Nodes = Nodes[Indices]

    # Sort elements according to their number
    Indices = np.argsort(Elements)
    Elements = Elements[Indices]
    ElementsNodes = ElementsNodes[Indices]
    Materials = Materials[Indices]

    # Identify top and bottom nodes
    Update(2/6,'Find top-bottom')
    Margin = 0.0
    BottomNodes = [Node[0] + 1 for Node in np.argwhere(Coords[:,2] <= Coords[:,2].min()+Margin)]
    TopNodes = [Node[0] + 1 for Node in np.argwhere(Coords[:,2] >= Coords[:,2].max()-Margin)]

    # Identify top and bottom elements
    BottomElements = np.isin(ElementsNodes,Nodes[np.array(BottomNodes)-1])
    BottomElements = [Element[0] + 1 for Element in np.argwhere(BottomElements.any(axis=1))]
    TopElements = np.isin(ElementsNodes,Nodes[np.array(TopNodes)-1])
    TopElements = [Element[0] + 1 for Element in np.argwhere(TopElements.any(axis=1))]

    # Compute boundary conditions
    Displacement = round((Coords[:,2].max() - Coords[:,2].min()) * Compression,3)

    # Generate nodes text
    Update(3/6,'Write nodes text')
    NodesStr = (np.arange(len(Nodes)) + 1).astype('<U32')
    CoordsStr = Coords.astype('<U32')
    Sep = np.array([', ', '\n']).astype('<U32')

    Ns = np.array_split(NodesStr,NProc)
    Cs = np.array_split(CoordsStr, NProc)
    Processes = [[N, C, Sep] for N, C in zip(Ns, Cs)]
    NText = []
    with Pool(processes=NProc) as P:
        for TaskResult in P.map(NodesText, Processes):
            NText.append(TaskResult)
    NText = ''.join(NText)

    # Generate element text
    Update(4/6,'Write elem. text')
    EN = (np.arange(len(Elements)) + 1).astype('<U32')
    NS = np.argsort(Nodes)
    SM = np.searchsorted(Nodes[NS], ElementsNodes)
    SEN = (NS[SM] + 1).astype('<U32')

    ENs = np.array_split(EN,NProc)
    SENs = np.array_split(SEN, NProc)
    Processes = [[E, S, Sep] for E, S in zip(ENs, SENs)]
    EText = []
    with Pool(processes=NProc) as P:
        for TaskResult in P.map(ElementsText, Processes):
            EText.append(TaskResult)
    EText = ''.join(EText)

    # Write file
    Update(5/6,'Write input file')
    with open(FileName,'w') as File:

        # Write heading
        File.write('*Heading\n')
        File.write('** Job name: ' + FileName[-Last:] + ' Model name: Bone\n')
        File.write('** Generated by: Python\n')
        File.write('*Preprint, echo=NO, model=NO, history=NO, contact=NO\n')
        File.write('**\n')

        # Write parts
        File.write('** PARTS\n')
        File.write('*Part, name=SAMPLE\n')

        # Write nodes
        File.write('**\n')
        File.write('*Node\n')
        File.write(NText)

        # Write elements
        File.write('**\n')
        File.write('*Element, type=C3D8\n')
        File.write(EText)

        # Write node set
        File.write('**\n')
        File.write('*Nset, nset=NODESET, generate\n')
        File.write('1,  ' + str(len(Coords)) + ',  1\n')

        # Write bone elements set
        File.write('**\n')
        File.write('*Elset, elset=BONE\n')
        Line = ''
        N = 0
        for Index, Material in enumerate(Materials):
            if Material == 255:
                Index += 1
                N += 1
                if np.mod((N),16) == 0:
                    File.write(Line[:-2] + '\n')
                    Line = str(Index) + ', '
                else:
                    Line += str(Index) + ', '
        File.write(Line[:-2] + '\n')

        # Write caps element set
        if Cap:
            File.write('**\n')
            File.write('*Elset, elset=CAPS\n')
            Line = ''
            N = 0
            for Index, Material in enumerate(Materials):
                if Material == 125:
                    Index += 1
                    N += 1
                    if np.mod((N),16) == 0:
                        File.write(Line[:-2] + '\n')
                        Line = str(Index) + ', '
                    else:
                        Line += str(Index) + ', '
            File.write(Line[:-2] + '\n')

        # Write section
        File.write('**\n')
        File.write('*Solid Section, elset=BONE, material=' + UMAT + '\n')
        if Cap:
            File.write('*Solid Section, elset=CAPS, material=Steel\n')
        File.write('*End Part\n')

        # Write assembly
        File.write('**\n')
        File.write('** ASSEMBLY\n')
        File.write('*Assembly, name=Assembly\n')

        ## Write instance
        File.write('**\n')
        File.write('*Instance, name=SAMPLE, part=SAMPLE\n')
        File.write('*End Instance\n')

        # Write bottom nodes set
        File.write('**\n')
        File.write('*Nset, nset=BOTTOMNODES, instance=SAMPLE\n')
        Line = ''
        for iBN, BN in enumerate(BottomNodes):
            if np.mod((iBN+1),16) == 0:
                File.write(Line[:-2] + '\n')
                Line = str(BN) + ', '
            else:
                Line += str(BN) + ', '
        File.write(Line[:-2] + '\n')

        # Write bottom elements set
        File.write('**\n')
        File.write('*Elset, elset=BOTTOMELEMENTS, instance=SAMPLE\n')
        Line = ''
        for iBE, BE in enumerate(BottomElements):
            if np.mod((iBE+1),16) == 0:
                File.write(Line[:-2] + '\n')
                Line = str(BE) + ', '
            else:
                Line += str(BE) + ', '
        File.write(Line[:-2] + '\n')

        # Write top nodes set
        File.write('**\n')
        File.write('*Nset, nset=TOPNODES, instance=SAMPLE\n')
        Line = ''
        for iTN, TN in enumerate(TopNodes):
            if np.mod((iTN+1),16) == 0:
                File.write(Line[:-2] + '\n')
                Line = str(TN) + ', '
            else:
                Line += str(TN) + ', '
        File.write(Line[:-2] + '\n')

        # Write top elements set
        File.write('**\n')
        File.write('*Elset, elset=TOPELEMENTS, instance=SAMPLE\n')
        Line = ''
        for iTE, TE in enumerate(TopElements):
            if np.mod((iTE+1),16) == 0:
                File.write(Line[:-2] + '\n')
                Line = str(TE) + ', '
            else:
                Line += str(TE) + ', '
        File.write(Line[:-2] + '\n')

        File.write('**\n')
        File.write('*End Assembly\n')

        # Write materials
        File.write('**\n')
        File.write('** MATERIALS\n')
        File.write('*Material, name=' + UMAT + '\n')
        File.write('*User Material, constants=2\n')
        File.write(str(E) + ', ' + str(Nu) + '\n')

        # Write dependent variables
        File.write('*DEPVAR\n')
        File.write('9\n')
        File.write('1, F11,\n')
        File.write('2, F12,\n')
        File.write('3, F13,\n')
        File.write('4, F21,\n')
        File.write('5, F22,\n')
        File.write('6, F23,\n')
        File.write('7, F31,\n')
        File.write('8, F32,\n')
        File.write('9, F33\n')
        File.write('**\n')

        if Cap:
            File.write('*Material, name=Steel\n')
            File.write('*ELASTIC\n')
            File.write('100000.0, 0.45\n')
            File.write('**\n')

        # Write step
        File.write('** STEP: Compression\n')
        File.write('*Step, name=Compression, nlgeom=YES\n')     # nlgeom=YES for F different from identity
        File.write('Static compression\n')
        File.write('*Static\n')
        File.write('1., 1., 1e-05, 1.\n')

        ## Write boundary conditions
        File.write('**\n')
        File.write('** BOUNDARY CONDITIONS\n')
        File.write('*Boundary\n')
        File.write('BOTTOMNODES, 1, 3, 0\n')
        File.write('*Boundary\n')
        File.write('TOPNODES, 1, 2, 0\n')
        File.write('*Boundary\n')
        File.write('TOPNODES, 3, 3, -' + str(Displacement) + '\n')

        ## Write output requests
        File.write('**\n')
        File.write('** OUTPUT REQUESTS\n')
        File.write('*Restart, write, frequency=0\n')
        File.write('** FIELD OUTPUT: F-Output-1\n')
        File.write('**\n')
        File.write('*Output, field\n')
        File.write('*Element Output\n')
        File.write('S,\n')
        File.write('SDV1,\n')
        File.write('SDV2,\n')
        File.write('SDV3,\n')
        File.write('SDV4,\n')
        File.write('SDV5,\n')
        File.write('SDV6,\n')
        File.write('SDV7,\n')
        File.write('SDV8,\n')
        File.write('SDV9,\n')
        File.write('*Node Output\n')
        File.write('U,\n')
        File.write('RF\n')
        File.write('**El Print\n')
        File.write('**DG\n')
        File.write('*Output, history, frequency=0\n')
        File.write('*End Step\n')

    Process(0,Text)

    return

def RunSim(self, WorkingDir, InputFile, UMAT, nCPUs=12):

    Text = 'Run ' + InputFile
    Process(1, Text)

    # Set working directory
    os.chdir(WorkingDir)

    # Absolutely necessary to start abaqus job
    try:
        os.environ.pop('PYTHONIOENCODING')
    except KeyError:
        pass

    # Write script
    Script = '''
#!/bin/bash
abaqus interactive job={Job} inp={InputFile} user={UMAT} cpus={nCPUs} ask_delete=OFF
'''
    
    Context = {'Job':InputFile[:-4],
            'InputFile':InputFile,
            'UMAT':UMAT,
            'nCPUs':nCPUs}

    FileName = 'Run.sh'
    with open(FileName, 'w') as File:
        File.write(Script.format(**Context))

    # Run simulation with script
    try:
        sh.bash(str(Path(WorkingDir, FileName))) # type: ignore
        Completed = True
    except:
        Completed = False
        print('Analysis not completed')

    if Completed:
        # Remove unnecessary files
        os.remove(InputFile[:-4] + '.com')
        os.remove(InputFile[:-4] + '.msg')
        os.remove(InputFile[:-4] + '.dat')
        os.remove(InputFile[:-4] + '.prt')
        os.remove(InputFile[:-4] + '.sim')
        os.remove(InputFile[:-4] + '.sta')

    # Print time
    Process(0)
    return

def ReadODB(self, WorkingDir, OdbFile, Variables=['U','RF'], Step=False, Frame=False, NodeSet=False):

    # Change working directory
    os.chdir(WorkingDir)

    # Write odb reader
    with open('ReadOdb.py','w') as File:

        # Write heading and initial part
        Text = """# ReadODB.py
# A script to read deformation gradient from odb file from ABAQUS.

import csv
import sys
import numpy as np
from odbAccess import *

#print \'Open odb file\'

Odb = openOdb(\'{File}\')
"""

        File.write(Text.format(**{'File':OdbFile + '.odb'}))

        if Step:
            Text = """
# Create variable that refers to the last frame of the step.
Steps = Odb.steps.keys()
Step = Steps[{Step}]
"""
            File.write(Text.format(**{'Step':Step}))

        else:
            Text = """
# Create variable that refers to the last frame of the step.
Steps = Odb.steps.keys()
Step = Steps[len(Steps)-1]
"""
            File.write(Text)

        if Frame:
            Text = """
Frame = Odb.steps[Step].frames[{Frame}]
"""
            File.write(Text.format(**{'Frame':Frame}))
            
        else:
            Text = """
Frames = Odb.steps[Step].frames
Shift = 1
while len(Frames) == 0:
Step = Steps[len(Steps)-(1+Shift)]
Frames = Odb.steps[Step].frames
Shift += 1
Frame = Odb.steps[Step].frames[len(Frames)-1]
"""
            File.write(Text)
            
        # Select instance
        Text ="""
# Create variable refering to model instance
Instances = Odb.rootAssembly.instances.keys()
Instance = Odb.rootAssembly.instances[Instances[0]]

"""
        File.write(Text)
                
        # Select fields outputs
        if 'U' in Variables:
            File.write('Displacements = Frame.fieldOutputs[\'U\'].values\n')
            
        if 'RF' in Variables:
            File.write('Forces = Frame.fieldOutputs[\'RF\'].values\n')
            
        if 'F' in Variables:
            File.write("""F11 = Frame.fieldOutputs['SDV_F11']
F12 = Frame.fieldOutputs['SDV_F12']
F13 = Frame.fieldOutputs['SDV_F13']
F21 = Frame.fieldOutputs['SDV_F21']
F22 = Frame.fieldOutputs['SDV_F22']
F23 = Frame.fieldOutputs['SDV_F23']
F31 = Frame.fieldOutputs['SDV_F31']
F32 = Frame.fieldOutputs['SDV_F32']
F33 = Frame.fieldOutputs['SDV_F33']

F = [F11,F12,F13,F21,F22,F23,F31,F32,F33]
F_Names = ['F11','F12','F13','F21','F22','F23','F31','F32','F33']
""")

        # Select nodes sets
        if NodeSet:
            Line = """\nNodes = odb.rootAssembly.nodeSets['{NodeSet}'].nodes[0]\n"""
            File.write(Line.format(**{'NodeSet':NodeSet}))
        else:
            File.write('Nodes = Instance.nodes\n')

        # Store nodes values
        if 'U' in Variables or 'RF' in Variables:
            File.write("""NodesData = []
for Node in Nodes:
Cx = float(Node.coordinates[0])
Cy = float(Node.coordinates[1])
Cz = float(Node.coordinates[2])
""")
            Line = '    NodesData.append([Cx, Cy, Cz'

            if 'U' in Variables:
                File.write("""    Ux = float(Displacements[Node.label].data[0])
Uy = float(Displacements[Node.label].data[1])
Uz = float(Displacements[Node.label].data[2])
""")
                Line += ', Ux, Uy, Uz'

            if 'RF' in Variables:
                File.write("""    Fx = float(Forces[Node.label].data[0])
Fy = float(Forces[Node.label].data[1])
Fz = float(Forces[Node.label].data[2])
""")
                Line += ', Fx, Fy, Fz'
            
            File.write(Line + '])\n')
            File.write('NodesData = np.array(NodesData)\n')

        # For deformation gradient analysis (to clean/adapt)
        if 'F' in Variables:
            File.write(r"""NodeLabels = np.array([])
for iNode, Node in enumerate(Nodes):
NodeLabels = np.append(NodeLabels,Node.label)

# Initialize loop
N = len(Instance.elements)
ElementsDG = np.zeros((12,N))

for ElementNumber, Element in enumerate(Instance.elements[:N]):
sys.stdout.write("\r" + "El. " + str(ElementNumber+1) + "/" + str(N))
sys.stdout.flush()

# Compute element central position
XYZ = np.zeros((len(Element.connectivity),3))
NodeNumber = 0
for NodeLabel in Element.connectivity:
    Node = np.where(NodeLabels==NodeLabel)[0][0]
    for Axis in range(3):
        XYZ[NodeNumber,Axis] = Nodes[Node].coordinates[Axis]
    NodeNumber += 1

# Get element mean deformation gradient
F_IntegrationPoints = np.zeros((1,9))

for F_Component in range(9):
    F_Value = F[F_Component].getSubset(region=Element).values[0].data
    F_IntegrationPoints[0,F_Component] = F_Value

# Add data to arrays and increment
ElementsDG[:3,ElementNumber] = np.mean(XYZ,axis=0)
ElementsDG[3:,ElementNumber] = F_IntegrationPoints
""")

        # Write into csv file
        #File.write('\nprint  \'' + r'\n' + 'Save to csv\'')

        if 'U' in Variables or 'RF' in Variables:
            File.write("""
with open('{File}_Nodes.csv', 'w') as File:
Writer = csv.writer(File)
for Row in NodesData:
    Writer.writerow(Row)  
""".format(**{'File':OdbFile}))

        if 'F' in Variables:
            File.write("""
with open('Elements_DG.csv', 'w') as File:
Writer = csv.writer(File)
for Row in ElementsDG.T:
    Writer.writerow(Row)            
""")

        # Close odb file
        File.write('Odb.close()')

    # Run odb reader
    os.system('abaqus python ReadOdb.py')

    # Collect results
    Data = []
    if 'U' in Variables or 'RF' in Variables:
        ND = pd.read_csv(str(Path(WorkingDir, OdbFile + '_Nodes.csv')), header=None)
        if 'RF' not in Variables:
            ND.columns = ['Cx','Cy','Cz','Ux','Uy','Uz']
        if 'U' not in Variables:
            ND.columns = ['Cx','Cy','Cz','Fx','Fy','Fz']
        else:
            ND.columns = ['Cx','Cy','Cz','Ux','Uy','Uz','Fx','Fy','Fz']
        Data.append(ND)

    if 'F' in Variables:
        DG = pd.read_csv('Elements_DG.csv', header=None)
        Data.append(DG)

    return Data

def ODB2VTK(self, WorkingDir, File):

    """
    Read odb file and write it into VTK to visualize with Paraview

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Adapted from: Qingbin Liu, Jiang Li, Jie Liu, 2017
                ParaView visualization of Abaqus output on the mechanical deformation of complex microstructures
                Computers and Geosciences, 99: 135-144
    
    March 2023
    """

    odbname = File                # Odb file name
    mesh_name = "Hexahedron"      # Element type
    mesh_corner = 8               # Nodes per element
    input_frame = range(0, 1 + 1) # Frames
    input_step = ['0']            # Step
    input_instance = ['0']        # Instance

    Dict = {'odbname':odbname,
            'mesh_name':mesh_name,
            'mesh_corner':mesh_corner,
            'input_frame':input_frame,
            'input_step':input_step,
            'input_instance':input_instance}

    Text = r"""from odbAccess import *
from textRepr import *
from string import *
from time import *

# Define frame, step and instance
input_frame = {input_frame}
input_step = {input_step}
input_instance = {input_instance}

# display the reading result of odb2vtk file
print('\nBasic Information:')
print('Model:' + '{odbname}' + '; Mesh type:' + '{mesh_name}' + '; Number of blocks:' + str(1))
print('Convert frames: ' + str(input_frame[0]) + ' to ' + str(input_frame[-1]))
print('Step & Instance : ' + str(input_step) + ', ' + str(input_instance))

# open an ODB ( Abaqus output database )
starttime = time()
odb = openOdb('{odbname}', readOnly=True)
print('\nODB opened')

# access geometry and topology information ( odb->rootAssembly->instances->(nodes, elements) )
rootassembly = odb.rootAssembly
instance = rootassembly.instances
# access attribute information
step = odb.steps
# get instance & step information : Quantity and all names
allinstancestr = str(instance)
autoins = allinstancestr.split("'")
inslen = len(autoins) / 4
instance_N = range(0, inslen)
allstepstr = str(step)
autostep = allstepstr.split("'")
steplen = len(autostep) / 4
step_N = range(0, steplen)

for i in input_step:
if (steplen < int(i)):
    print('\nInput step exceeds the range of steps')
    os._exit(0)
for i in input_instance:
if (inslen < int(i)):
    print('\nInput instance exceeds the range of instances')
    os._exit(0)

# step cycle
for step_i in input_step:
n = int(step_i) * 4 + 1
stepname = autostep[n]
print('\nStep: ' + stepname)
# instance cycle
for ins_i in input_instance:
    n = int(ins_i) * 4 + 1
    instancename = autoins[n]
    print('\nInstance: ' + instancename)

    # access nodes & elements
    node = instance[instancename].nodes
    element = instance[instancename].elements
    n_nodes = len(node)
    n_elements = len(element)
    # access attribute(fieldOutputs) information
    frame = step[stepname].frames

    # compute the number of element of each block
    p_elements = n_elements + 1
    lp_elements = n_elements # last block

    # match nodes' label and its order in sequence (for empty nodes in tetra mesh)
    MLN = node[n_nodes - 1].label
    TOTAL = []
    # read node in sequence, and get the largest label of node(non-empty)
    # MLN is the max label of nodeset
    for i in node:
        TOTAL.append(i.label)
        if (i.label > MLN):
            MLN = i.label
    # match (the key)
    L = []
    n = 0
    for i in range(MLN):
        L.append(0)
    for i in TOTAL:
        L[i - 1] = n
        n += 1

    # frame cycle
    for i_frame in input_frame:

        # Detect whether the input frame is out of range
        try:
            TRY = odb.steps[stepname].frames[int(i_frame)]
        except:
            print('\nInput frame exceeds the range of frames')
            os._exit(0)
            break

        # Access a frame
        N_Frame = odb.steps[stepname].frames[int(i_frame)]
        print('\nFrame:' + str(i_frame))

        # create array for store result data temporarily
        # Vector-U,A,V,RF
        L0 = []
        # Tensors-S
        L1 = []
        for i in range(MLN):
            L0.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            L1.append([0, 0])

        print('\nReading U, RF ...')
        time1 = time()
        # Access Spatial displacement
        displacement = N_Frame.fieldOutputs['U']
        fieldValues = displacement.values
        for valueX in fieldValues:
            i = valueX.nodeLabel
            L0[i - 1][0] = valueX.data[0]
            L0[i - 1][1] = valueX.data[1]
            L0[i - 1][2] = valueX.data[2]

        # Access Reaction force
        Reaction_force = N_Frame.fieldOutputs['RF']
        fieldValues = Reaction_force.values
        for valueX in fieldValues:
            i = valueX.nodeLabel
            L0[i - 1][9] = valueX.data[0]
            L0[i - 1][10] = valueX.data[1]
            L0[i - 1][11] = valueX.data[2]
        print('Time elapsed: %.3f s' % (time() - time1))
        print('\nReading Stress ...')
        time1 = time()
        # access Stress components
        Stress = N_Frame.fieldOutputs['S']
        node_Stress = Stress.getSubset(position=ELEMENT_NODAL)
        fieldValues = node_Stress.values
        for valueX in fieldValues:
            L1[valueX.nodeLabel - 1][0] += 1
            L1[valueX.nodeLabel - 1][1] += valueX.mises
        # can first ave
        print('Time elapsed: %.3f s' % (time() - time1))

        '''============================================================'''

        print('\nPartitionning model and writing vtk files ...')
        time1 = time()
        print('Frame:' + str(i_frame) + '; Block:' + str(0))
        
        # Reorganization
        # Control&Storage
        # estimate whether the node has already existed
        stg_p = []
        
        # store the reorganized node for element
        stg_e = []
        
        # store the reorganized node for node
        stg_n = []
        for i in range(MLN):
            stg_p.append(-1)
        nodecount = 0
        
        # reorganize the node and element (reconstruct the mesh)
        for i in range(n_elements):
            for j in range({mesh_corner}):
                k = element[i].connectivity[j] - 1
                if (stg_p[k] < 0):
                    stg_p[k] = nodecount
                    stg_n.append(L[k])
                    stg_e.append(nodecount)
                    nodecount += 1
                else:
                    stg_e.append(stg_p[k])
                    
        # compute point quantity
        n_reop = len(stg_n)
        reop_N = range(0, len(stg_n))

        # create and open a VTK(.vtu) files
        outfile = open('{odbname}'[:-4] + '_' + stepname + '_' + instancename + 'f%03d' % int(i_frame) + '.vtu', 'w')

        # <VTKFile>, including the type of mesh, version, and byte_order
        outfile.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">' + '\n')
        
        # <UnstructuredGrid>
        outfile.write('<UnstructuredGrid>' + '\n')
        
        # <Piece>, including the number of points and cells
        outfile.write('<Piece NumberOfPoints="' + str(n_reop) + '"' + ' ' + 'NumberOfCells="' + str(
            lp_elements) + '">' + '\n')

        print('Writing Nodes ...')
        # <Points> Write nodes into vtk files
        displacement = N_Frame.fieldOutputs['U']
        fieldValues = displacement.values
        outfile.write('<Points>' + '\n')
        outfile.write('<DataArray type="Float64" NumberOfComponents="3" format="ascii">' + '\n')
        for i in reop_N:
            nt = stg_n[i]
            k = node[stg_n[i]].label - 1
            X, Y, Z = node[nt].coordinates[0] + L0[k][0], node[nt].coordinates[1] + L0[k][1], \
                        node[nt].coordinates[2] + L0[k][2]
            outfile.write(' ' + '%11.8e' % X + '  ' + '%11.8e' % Y + '  ' + '%11.8e' % Z + '\n')
        outfile.write('</DataArray>' + '\n')
        outfile.write('</Points>' + '\n')
        # </Points>

        print('Writing Results data ...')
        # <PointData> Write results data into vtk files
        outfile.write("<" + "PointData" + " " + "Vectors=" + '"' + "Spatial_displacement,Reaction_force" + '"' \
                        + " " + "Scalars=" + '"' + "Stress_Mises" + '"' + ">" + '\n')

        # Spatial displacement, <DataArray>
        outfile.write(
            "<" + "DataArray" + " " + "type=" + '"' + "Float32" + '"' + " " + "Name=" + '"' + "Spatial_displacement" + '"' + " " + "NumberOfComponents=" + '"' + "3" + '"' + " " + "format=" + '"' + "ascii" + '"' + ">" + '\n')
        for i in reop_N:
            k = node[stg_n[i]].label - 1
            X, Y, Z = L0[k][0], L0[k][1], L0[k][2]
            outfile.write('%11.8e' % X + ' ' + '%11.8e' % Y + ' ' + '%11.8e' % Z + '\n')
        outfile.write("</DataArray>" + '\n')
        # </DataArray>

        # Reaction force
        outfile.write(
            "<" + "DataArray" + " " + "type=" + '"' + "Float32" + '"' + " " + "Name=" + '"' + "Reaction_force" + '"' + " " + "NumberOfComponents=" + '"' + "3" + '"' + " " + "format=" + '"' + "ascii" + '"' + ">" + '\n')
        for i in reop_N:
            k = node[stg_n[i]].label - 1
            X, Y, Z = L0[k][9], L0[k][10], L0[k][11]
            outfile.write('%11.8e' % X + ' ' + '%11.8e' % Y + ' ' + '%11.8e' % Z + '\n')
        outfile.write("</DataArray>" + '\n')
        # </DataArray>

        # Stress Mises, <DataArray>
        outfile.write(
            "<" + "DataArray" + " " + "type=" + '"' + "Float32" + '"' + " " + "Name=" + '"' + "Stress_Mises" + '"' + " " + "format=" + '"' + "ascii" + '"' + ">" + '\n')
        for i in reop_N:
            k = node[stg_n[i]].label - 1
            X = L1[k][1] / L1[k][0]
            outfile.write('%11.8e' % X + '\n')
        outfile.write('</DataArray>' + '\n')
        # </DataArray>

        outfile.write("</PointData>" + '\n')
        # </PointData>

        print('Writing Cells ...')
        # <Cells> Write cells into vtk files
        outfile.write('<Cells>' + '\n')
        # Connectivity
        outfile.write('<DataArray type="Int32" Name="connectivity" format="ascii">' + '\n')
        for i in range(len(stg_e) / 8):
            outfile.write(str(stg_e[i * 8]) + ' ' + str(stg_e[i * 8 + 1]) + ' ' + str(
                stg_e[i * 8 + 2]) + ' ' + str(stg_e[i * 8 + 3]) + ' ' + str(
                stg_e[i * 8 + 4]) + ' ' + str(stg_e[i * 8 + 5]) + ' ' + str(
                stg_e[i * 8 + 6]) + ' ' + str(stg_e[i * 8 + 7]) + '\n')
        outfile.write('</DataArray>' + '\n')
        
        # Offsets
        outfile.write('<DataArray type="Int32" Name="offsets" format="ascii">' + '\n')
        for i in range(len(stg_e) / {mesh_corner}):
            outfile.write(str(i * {mesh_corner} + {mesh_corner}) + '\n')
        outfile.write('</DataArray>' + '\n')
        
        # Type
        outfile.write('<DataArray type="UInt8" Name="types" format="ascii">' + '\n')
        for i in range(len(stg_e) / {mesh_corner}):
            outfile.write(str(12) + '\n')
        outfile.write('</DataArray>' + '\n')
        outfile.write('</Cells>' + '\n')
        # </Cells>

        # </Piece>
        outfile.write('</Piece>' + '\n')
        # </UnstructuredGrid>
        outfile.write('</UnstructuredGrid>' + '\n')
        # </VTKFile>
        outfile.write('</VTKFile>' + '\n')

        outfile.close()
        print('Time elapsed: %.3f s' % (time() - time1))

        '''====================================================================='''
        print('\nCreating .pvtu file for frame ' + str(i_frame) + ' ...')

    odb.close()

print('Total time elapsed: %.3f s' % (time() - starttime))
"""

    # Write and execute ODB to VTK
    os.chdir(WorkingDir)

    with open('Odb2Vtk.py','w') as File:
        File.write(Text.format(**Dict))

    os.system('abaqus python Odb2Vtk.py')

    return


@njit
def NumbaMapping(Array):

    X, Y, Z = Array.T.shape

    # Generate nodes map
    Index = 0
    Nodes = np.zeros((Z+1,Y+1,X+1),'int')
    Coords = np.zeros((Z+1,Y+1,X+1,3),'int')
    for Xn in range(X + 1):
        for Yn in range(Y + 1):
            for Zn in range(Z + 1):
                Index += 1
                Nodes[Zn,Yn,Xn] = Index
                Coords[Zn,Yn,Xn] = [Xn, Yn, Zn]
    Nodes = Nodes[::-1,::-1]
    Coords = Coords[:,::-1]

    # Generate elements map
    Index = 0
    Elements = np.zeros((Z, Y, X),'int')
    ElementsNodes = np.zeros((Z, Y, X, 8), 'int')
    for Xn in range(X):
            for Yn in range(Y):
                for Zn in range(Z):
                    Index += 1
                    Elements[Zn, Yn, Xn] = Index
                    ElementsNodes[Zn, Yn, Xn, 0] = Nodes[Zn+1, Yn, Xn]
                    ElementsNodes[Zn, Yn, Xn, 1] = Nodes[Zn+1, Yn+1, Xn]
                    ElementsNodes[Zn, Yn, Xn, 2] = Nodes[Zn, Yn+1, Xn]
                    ElementsNodes[Zn, Yn, Xn, 3] = Nodes[Zn, Yn, Xn]
                    ElementsNodes[Zn, Yn, Xn, 4] = Nodes[Zn+1, Yn, Xn+1]
                    ElementsNodes[Zn, Yn, Xn, 5] = Nodes[Zn+1, Yn+1, Xn+1]
                    ElementsNodes[Zn, Yn, Xn, 6] = Nodes[Zn, Yn+1, Xn+1]
                    ElementsNodes[Zn, Yn, Xn, 7] = Nodes[Zn, Yn, Xn+1]
    Elements = Elements[::-1, ::-1]

    return Nodes, Coords, Elements, ElementsNodes

#@njit
def NodesText(NCS):
    Nodes, Coords, Sep = NCS
    NText = [''.join([N, Sep[0],
                    C[0], Sep[0],
                    C[1], Sep[0],
                    C[2], Sep[1]]) for N, C in zip(Nodes, Coords)]
    NText = ''.join(NText)
    return NText

#@njit
def ElementsText(ESS):
    ElementsNumber, SortedElementsNodes, Sep = ESS
    EText = [''.join([N, Sep[0],
                    E[0], Sep[0],
                    E[1], Sep[0],
                    E[2], Sep[0],
                    E[3], Sep[0],
                    E[4], Sep[0],
                    E[5], Sep[0],
                    E[6], Sep[0],
                    E[7], Sep[1]]) for N, E in zip(ElementsNumber, SortedElementsNodes)]
    EText = ''.join(EText)

    return EText


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
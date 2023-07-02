#%% #!/usr/bin/env python3
# Initialization

Version = '01'

Description = """
    Script used for tensorial operations

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


#%% Functions
# Define functions

def __init__(self):
        pass

def CheckPosition(self, A, B):
    AShape = A.shape
    BShape = B.shape
    if AShape[len(AShape) - 1] < BShape[0]:
        print('\nInconsistent Shape  A=', AShape, ' B=', BShape)
    return

def CheckShape(self, A):
    Shape = A.shape
    for Index in range(len(Shape)):
        if Shape[Index] != 3:
            print('\nOrder of Tensor is not correct: ', Shape, '\n')

def CheckMinorSymmetry(self, A):
    MinorSymmetry = True
    for i in range(3):
        for j in range(3):
            PartialTensor = A[:,:, i, j]
            if PartialTensor[1, 0] == PartialTensor[0, 1] and PartialTensor[2, 0] == PartialTensor[0, 2] and PartialTensor[1, 2] == PartialTensor[2, 1]:
                MinorSymmetry = True
            else:
                MinorSymmetry = False
                break

    if MinorSymmetry == True:
        for i in range(3):
            for j in range(3):
                PartialTensor = np.squeeze(A[i, j,:,:])
                if PartialTensor[1, 0] == PartialTensor[0, 1] and PartialTensor[2, 0] == PartialTensor[0, 2] and PartialTensor[1, 2] == PartialTensor[2, 1]:
                    MinorSymmetry = True
                else:
                    MinorSymmetry = False
                    break

def Length(self, a):
    c = np.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    return c

def UnitVector(self, a):
    l = self.Length(a)
    c = a / l
    return c

def UnitMatrix(self, n):
    I = np.zeros((n, n), float)
    for row in range(n):
        for col in range(n):
            if row == col:
                I[col, row] = 1.0

    return I

def CrossProduct(self, a, b):
    c1 = a[1] * b[2] - a[2] * b[1]
    c2 = a[2] * b[0] - a[0] * b[2]
    c3 = a[0] * b[1] - a[1] * b[0]
    c = np.array([c1, c2, c3])
    return c

def DyadicProduct(self, A, B):

    self.CheckShape(A)
    self.CheckShape(B)
    self.CheckPosition(A, B)

    type = 10 * len(A.shape) + len(B.shape)
    C = np.array([])
    if type == 11:
        C = np.zeros((3, 3), float)
        for i in range(3):
            for j in range(3):
                C[i, j] = A[i] * B[j]

    elif type == 21:
        C = np.zeros((3, 3, 3), float)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    C[i, j, k] = A[i, j] * B[k]

    elif type == 22:
        C = np.zeros((3, 3, 3, 3), float)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        C[i, j, k, l] = A[i, j] * B[k, l]

    else:
        print('Dyadic product not supported')

    return C

def FrobeniusProduct(self, A, B):

    self.CheckShape(A)
    self.CheckShape(B)

    s = 0

    if A.size == 9 and B.size == 9:
        for i in range(3):
            for j in range(3):
                s += A[i, j] * B[i, j]

    elif A.size == 36 and B.size == 36:
        for i in range(6):
            for j in range(6):
                s = s + A[i, j] * B[i, j]

    elif A.shape == (9,9) and B.shape == (9,9):
        for i in range(9):
            for j in range(9):
                s = s + A[i, j] * B[i, j]

    elif A.shape == (3,3,3,3) and B.shape == (3,3,3,3):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        s = s + A[i, j, k, l] * B[i, j, k, l]

    else:
        print('Matrices sizes mismatch')

    return s

def SymmetricProduct(self, A, B):

    if isinstance(A[0, 0], np.float):
        C = np.zeros((3, 3, 3, 3))
    else:
        C = sp.zeros(9)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    if isinstance(A[0, 0], np.float):
                        C[i,j,k,l] = (1/2)*(A[i,k]*B[j,l]+A[i,l]*B[j,k])
                    else:
                        C[3*i+j,3*k+l] = (1/2)*(A[i,k]*B[j,l]+A[i,l]*B[j,k])

    if not isinstance(A[0,0], np.float):
        C = self.IsoMorphism99_3333(C)

    return C

def DoubleContraction(self, A, B):

    self.CheckShape(A)
    self.CheckShape(B)
    self.CheckPosition(A, B)

    type = 10 * len(A.shape) + len(B.shape)
    C = np.array([])
    if type == 22:
        C = np.zeros((1, ), float)
        for i in range(3):
            for j in range(3):
                C[(0, )] = C[(0, )] + A[i, j] * B[i, j]

    elif type == 42:
        C = np.zeros((3, 3), float)
        for i in range(3):
            for j in range(3):
                for m in range(3):
                    for n in range(3):
                        C[i, j] = C[i, j] + A[i, j, m, n] * B[m, n]

    elif type == 44:
        C = np.zeros((1, ), float)
        for i in range(3):
            for j in range(3):
                for m in range(3):
                    for n in range(3):
                        C[(0, )] = C[(0, )] + A[i, j, m, n] * B[i, j, m, n]

    else:
        print('Double contraction not supported')

    if C.shape[0] == 1:
        return C[0]
        
    else:
        return C

def Transform(self, A, B):

    self.CheckShape(A)
    self.CheckShape(B)

    if A.size == 9 and B.size == 3:

        if isinstance(A[0,0], np.float):
            c = np.zeros(3)
        else:
            c = sp.Matrix([0,0,0])

        for i in range(3):
            for j in range(3):
                c[i] += A[i,j] * B[j]

        if not isinstance(A[0,0], np.float):
            c = np.array(c)

        return c

    elif A.size == 27 and B.size == 9:

        if isinstance(A[0,0,0], np.float):
            c = np.zeros(3)
        else:
            c = sp.Matrix([0,0,0])

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    c[i] += A[i,j,k] * B[j,k]

        if not isinstance(A[0,0,0], np.float):
            c = np.array(c)

        return c

    elif A.size == 81 and B.size == 9:

        if isinstance(A[0,0,0,0], np.float):
            c = np.zeros((3,3))
        else:
            c = sp.zeros(3)

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        c[i,j] += A[i,j,k,l] * B[k,l]

        if not isinstance(A[0,0,0,0], np.float):
            c = np.array(c)

        return c

    else:
        print('Matrices sizes mismatch')

def IsoMorphism99_3333(self, A):

    if isinstance(A[0, 0], np.float):
        B = np.zeros((3, 3, 3, 3))

    else:
        B = sp.zeros(9)
        B = np.array(B).reshape((3,3,3,3))

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    B[i,j,k,l] = A[3*i+j,3*k+l]

    return B

def IsoMorphism3333_66(self, A):

    if self.CheckMinorSymmetry(A) == False:
        print('Tensor does not present minor symmetry')
    else:

        if isinstance(A[0, 0, 0, 0], np.float):
            B = np.zeros((6,6))
        else:
            B = sp.zeros(6)

        B[0, 0] = A[0, 0, 0, 0]
        B[1, 0] = A[1, 1, 0, 0]
        B[2, 0] = A[2, 2, 0, 0]
        B[3, 0] = np.sqrt(2) * A[1, 2, 0, 0]
        B[4, 0] = np.sqrt(2) * A[2, 0, 0, 0]
        B[5, 0] = np.sqrt(2) * A[0, 1, 0, 0]

        B[0, 1] = A[0, 0, 1, 1]
        B[1, 1] = A[1, 1, 1, 1]
        B[2, 1] = A[2, 2, 1, 1]
        B[3, 1] = np.sqrt(2) * A[1, 2, 1, 1]
        B[4, 1] = np.sqrt(2) * A[2, 0, 1, 1]
        B[5, 1] = np.sqrt(2) * A[0, 1, 1, 1]

        B[0, 2] = A[0, 0, 2, 2]
        B[1, 2] = A[1, 1, 2, 2]
        B[2, 2] = A[2, 2, 2, 2]
        B[3, 2] = np.sqrt(2) * A[1, 2, 2, 2]
        B[4, 2] = np.sqrt(2) * A[2, 0, 2, 2]
        B[5, 2] = np.sqrt(2) * A[0, 1, 2, 2]

        B[0, 3] = np.sqrt(2) * A[0, 0, 1, 2]
        B[1, 3] = np.sqrt(2) * A[1, 1, 1, 2]
        B[2, 3] = np.sqrt(2) * A[2, 2, 1, 2]
        B[3, 3] = 2 * A[1, 2, 1, 2]
        B[4, 3] = 2 * A[2, 0, 1, 2]
        B[5, 3] = 2 * A[0, 1, 1, 2]

        B[0, 4] = np.sqrt(2) * A[0, 0, 2, 0]
        B[1, 4] = np.sqrt(2) * A[1, 1, 2, 0]
        B[2, 4] = np.sqrt(2) * A[2, 2, 2, 0]
        B[3, 4] = 2 * A[1, 2, 2, 0]
        B[4, 4] = 2 * A[2, 0, 2, 0]
        B[5, 4] = 2 * A[0, 1, 2, 0]

        B[0, 5] = np.sqrt(2) * A[0, 0, 0, 1]
        B[1, 5] = np.sqrt(2) * A[1, 1, 0, 1]
        B[2, 5] = np.sqrt(2) * A[2, 2, 0, 1]
        B[3, 5] = 2 * A[1, 2, 0, 1]
        B[4, 5] = 2 * A[2, 0, 0, 1]
        B[5, 5] = 2 * A[0, 1, 0, 1]

        return B

def IsoMorphism66_3333(self, A):

    # Check symmetry
    Symmetry = True
    for i in range(6):
        for j in range(6):
            if not A[i,j] == A[j,i]:
                Symmetry = False
                break
    if Symmetry == False:
        print('Matrix is not symmetric!')
        return

    if isinstance(A[0, 0], np.float):
        B = np.zeros((3,3,3,3))
    else:
        B = sp.zeros((3,3,3,3))

    # Build 4th tensor
    B[0, 0, 0, 0] = A[0, 0]
    B[1, 1, 0, 0] = A[1, 0]
    B[2, 2, 0, 0] = A[2, 0]
    B[1, 2, 0, 0] = A[3, 0] / np.sqrt(2)
    B[2, 0, 0, 0] = A[4, 0] / np.sqrt(2)
    B[0, 1, 0, 0] = A[5, 0] / np.sqrt(2)

    B[0, 0, 1, 1] = A[0, 1]
    B[1, 1, 1, 1] = A[1, 1]
    B[2, 2, 1, 1] = A[2, 1]
    B[1, 2, 1, 1] = A[3, 1] / np.sqrt(2)
    B[2, 0, 2, 1] = A[4, 1] / np.sqrt(2)
    B[0, 1, 2, 1] = A[5, 1] / np.sqrt(2)

    B[0, 0, 2, 2] = A[0, 2]
    B[1, 1, 2, 2] = A[1, 2]
    B[2, 2, 2, 2] = A[2, 2]
    B[1, 2, 2, 2] = A[3, 2] / np.sqrt(2)
    B[2, 0, 2, 2] = A[4, 2] / np.sqrt(2)
    B[0, 1, 2, 2] = A[5, 2] / np.sqrt(2)

    B[0, 0, 1, 2] = A[0, 3] / np.sqrt(2)
    B[1, 1, 1, 2] = A[1, 3] / np.sqrt(2)
    B[2, 2, 1, 2] = A[2, 3] / np.sqrt(2)
    B[1, 2, 1, 2] = A[3, 3] / 2
    B[2, 0, 1, 2] = A[4, 3] / 2
    B[0, 1, 1, 2] = A[5, 3] / 2

    B[0, 0, 2, 0] = A[0, 4] / np.sqrt(2)
    B[1, 1, 2, 0] = A[1, 4] / np.sqrt(2)
    B[2, 2, 2, 0] = A[2, 4] / np.sqrt(2)
    B[1, 2, 2, 0] = A[3, 4] / 2
    B[2, 0, 2, 0] = A[4, 4] / 2
    B[0, 1, 2, 0] = A[5, 4] / 2

    B[0, 0, 0, 1] = A[0, 5] / np.sqrt(2)
    B[1, 1, 0, 1] = A[1, 5] / np.sqrt(2)
    B[2, 2, 0, 1] = A[2, 5] / np.sqrt(2)
    B[1, 2, 0, 1] = A[3, 5] / 2
    B[2, 0, 0, 1] = A[4, 5] / 2
    B[0, 1, 0, 1] = A[5, 5] / 2



    # Add minor symmetries ijkl = ijlk and ijkl = jikl

    B[0, 0, 0, 0] = B[0, 0, 0, 0]
    B[0, 0, 0, 0] = B[0, 0, 0, 0]

    B[0, 0, 1, 0] = B[0, 0, 0, 1]
    B[0, 0, 0, 1] = B[0, 0, 0, 1]

    B[0, 0, 1, 1] = B[0, 0, 1, 1]
    B[0, 0, 1, 1] = B[0, 0, 1, 1]

    B[0, 0, 2, 1] = B[0, 0, 1, 2]
    B[0, 0, 1, 2] = B[0, 0, 1, 2]

    B[0, 0, 2, 2] = B[0, 0, 2, 2]
    B[0, 0, 2, 2] = B[0, 0, 2, 2]

    B[0, 0, 0, 2] = B[0, 0, 2, 0]
    B[0, 0, 2, 0] = B[0, 0, 2, 0]



    B[0, 1, 0, 0] = B[0, 1, 0, 0]
    B[1, 0, 0, 0] = B[0, 1, 0, 0]

    B[0, 1, 1, 0] = B[0, 1, 0, 1]
    B[1, 0, 0, 1] = B[0, 1, 0, 1]

    B[0, 1, 1, 1] = B[0, 1, 1, 1]
    B[1, 0, 1, 1] = B[0, 1, 1, 1]

    B[0, 1, 2, 1] = B[0, 1, 1, 2]
    B[1, 0, 1, 2] = B[0, 1, 1, 2]

    B[0, 1, 2, 2] = B[0, 1, 2, 2]
    B[1, 0, 2, 2] = B[0, 1, 2, 2]

    B[0, 1, 0, 2] = B[0, 1, 2, 0]
    B[1, 0, 2, 0] = B[0, 1, 2, 0]



    B[1, 1, 0, 0] = B[1, 1, 0, 0]
    B[1, 1, 0, 0] = B[1, 1, 0, 0]

    B[1, 1, 1, 0] = B[1, 1, 0, 1]
    B[1, 1, 0, 1] = B[1, 1, 0, 1]

    B[1, 1, 1, 1] = B[1, 1, 1, 1]
    B[1, 1, 1, 1] = B[1, 1, 1, 1]

    B[1, 1, 2, 1] = B[1, 1, 1, 2]
    B[1, 1, 1, 2] = B[1, 1, 1, 2]

    B[1, 1, 2, 2] = B[1, 1, 2, 2]
    B[1, 1, 2, 2] = B[1, 1, 2, 2]

    B[1, 1, 0, 2] = B[1, 1, 2, 0]
    B[1, 1, 2, 0] = B[1, 1, 2, 0]



    B[1, 2, 0, 0] = B[1, 2, 0, 0]
    B[2, 1, 0, 0] = B[1, 2, 0, 0]

    B[1, 2, 1, 0] = B[1, 2, 0, 1]
    B[2, 1, 0, 1] = B[1, 2, 0, 1]

    B[1, 2, 1, 1] = B[1, 2, 1, 1]
    B[2, 1, 1, 1] = B[1, 2, 1, 1]

    B[1, 2, 2, 1] = B[1, 2, 1, 2]
    B[2, 1, 1, 2] = B[1, 2, 1, 2]

    B[1, 2, 2, 2] = B[1, 2, 2, 2]
    B[2, 1, 2, 2] = B[1, 2, 2, 2]

    B[1, 2, 0, 2] = B[1, 2, 2, 0]
    B[2, 1, 2, 0] = B[1, 2, 2, 0]



    B[2, 2, 0, 0] = B[2, 2, 0, 0]
    B[2, 2, 0, 0] = B[2, 2, 0, 0]

    B[2, 2, 1, 0] = B[2, 2, 0, 1]
    B[2, 2, 0, 1] = B[2, 2, 0, 1]

    B[2, 2, 1, 1] = B[2, 2, 1, 1]
    B[2, 2, 1, 1] = B[2, 2, 1, 1]

    B[2, 2, 2, 1] = B[2, 2, 1, 2]
    B[2, 2, 1, 2] = B[2, 2, 1, 2]

    B[2, 2, 2, 2] = B[2, 2, 2, 2]
    B[2, 2, 2, 2] = B[2, 2, 2, 2]

    B[2, 2, 0, 2] = B[2, 2, 2, 0]
    B[2, 2, 2, 0] = B[2, 2, 2, 0]



    B[2, 0, 0, 0] = B[2, 0, 0, 0]
    B[0, 2, 0, 0] = B[2, 0, 0, 0]

    B[2, 0, 1, 0] = B[2, 0, 0, 1]
    B[0, 2, 0, 1] = B[2, 0, 0, 1]

    B[2, 0, 1, 1] = B[2, 0, 1, 1]
    B[0, 2, 1, 1] = B[2, 0, 1, 1]

    B[2, 0, 2, 1] = B[2, 0, 1, 2]
    B[0, 2, 1, 2] = B[2, 0, 1, 2]

    B[2, 0, 2, 2] = B[2, 0, 2, 2]
    B[0, 2, 2, 2] = B[2, 0, 2, 2]

    B[2, 0, 0, 2] = B[2, 0, 2, 0]
    B[0, 2, 2, 0] = B[2, 0, 2, 0]


    # Complete minor symmetries
    B[0, 2, 1, 0] = B[0, 2, 0, 1]
    B[0, 2, 0, 2] = B[0, 2, 2, 0]
    B[0, 2, 2, 1] = B[0, 2, 1, 2]

    B[1, 0, 1, 0] = B[1, 0, 0, 1]
    B[1, 0, 0, 2] = B[1, 0, 2, 0]
    B[1, 0, 2, 1] = B[1, 0, 1, 2]

    B[2, 1, 1, 0] = B[2, 1, 0, 1]
    B[2, 1, 0, 2] = B[2, 1, 2, 0]
    B[2, 1, 2, 1] = B[2, 1, 1, 2]


    # Add major symmetries ijkl = klij
    B[0, 1, 1, 1] = B[1, 1, 0, 1]
    B[1, 0, 1, 1] = B[1, 1, 1, 0]

    B[0, 2, 1, 1] = B[1, 1, 0, 2]
    B[2, 0, 1, 1] = B[1, 1, 2, 0]


    return B

# Rotate fabric (use this trick in tensor rotation)
# R = np.linalg.inv(Fabric[1])
# M = np.zeros((3,3))
# for i, m in enumerate(Fabric[0]):
#     M += m * np.outer(Fabric[1][:,i], Fabric[1][:,i])
# RM = np.dot(np.dot(R,M),Fabric[1])
# np.linalg.eigvals(RM) # check that eigen values didn't change


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
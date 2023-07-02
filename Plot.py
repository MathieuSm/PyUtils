#%% #!/usr/bin/env python3
# Initialization

Version = '01'

Description = """
    Script for general plotting functions

    Version Control:
        01 - Original script

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel, University of Bern

    Date: july 2023
    """

#%% Imports
# Modules import

import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from matplotlib.colors import ListedColormap
from scipy.stats.distributions import t, norm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from General import Resample
from Tensor import DyadicProduct, FrobeniusProduct, Transform

#%% Functions
# Define functions

def __init__(self):
        self.FName = None
        self.ShowPlot = True
        self.IRange = [0.8, 1.2]

def Normalize(self, Array, uint=False):

    Min = np.min(Array)
    Max = np.max(Array)
    N_Array = (Array - Min) / (Max - Min)

    if uint:
        N_Array = np.array(N_Array * 255).astype('uint8') 

    return N_Array

def ROI3D(self, Image, Color=None, Title=None, Angles=[30, 45, 15]):

    Array = sitk.GetArrayFromImage(Image)

    # If too big, interpolate image
    if Array.size > 1.5E5:
        Image = Resample(Image, Size=(50, 50, 50))
        Array = sitk.GetArrayFromImage(Image)

    # Get x, y, z coordinate
    Z, Y, X = np.where(Array)

    Figure = plt.figure(figsize=(5.5, 4))
    Axis = Figure.add_subplot(111, projection='3d')
    
    # scaling hack
    Bbox_min = np.min([X, Y, Z])
    Bbox_max = np.max([X, Y, Z])
    Axis.auto_scale_xyz([Bbox_min, Bbox_max], [Bbox_min, Bbox_max], [Bbox_min, Bbox_max])
    Axis.voxels(Array, facecolors=(227/255, 218/255, 201/255))
    
    # make the panes transparent
    Axis.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    Axis.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    Axis.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    # make the grid lines transparent
    Axis.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    Axis.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    Axis.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    Axis.set_xlabel('X')
    Axis.set_ylabel('Y')
    Axis.set_zlabel('Z')
    Axis.set_axis_off()

    if (Title):
        Axis.set_title(Title)

    # Rotate plot
    Axis.view_init(elev=Angles[0], azim=Angles[1])

    plt.show()

def Slice(self, Image, Slice=None, Title=None, Axis='Z'):

    try:
        Array = sitk.GetArrayFromImage(Image)
        Dimension = Image.GetDimension()
    except:
        Array = Image
        Dimension = len(Array.shape)

    if Dimension == 3:
        
        if Axis == 'Z':
            if Slice:
                Array = Array[Slice,:,:]
            else:
                Array = Array[Array.shape[0]//2,:,:]
        if Axis == 'Y':
            if Slice:
                Array = Array[:,Slice,:]
            else:
                Array = Array[:,Array.shape[1]//2,:]
        if Axis == 'X':
            if Slice:
                Array = Array[:,:,Slice]
            else:
                Array = Array[:,:,Array.shape[2]//2]

    Figure, Axis = plt.subplots()
    Axis.imshow(Array,interpolation=None, cmap='binary_r')
    Axis.axis('Off')
    
    if (Title):
        Axis.set_title(Title)

    if (self.FName):
        plt.savefig(self.FName, bbox_inches='tight', pad_inches=0)

    if self.ShowPlot:
        plt.show()
    else:
        plt.close()

    return

def Overlay(self, Fixed, Moving, Slice=None, Title=None, Axis='Z', AsBinary=False):

    FixedArray = sitk.GetArrayFromImage(Fixed)
    MovingArray = sitk.GetArrayFromImage(Moving)

    if AsBinary:
        Otsu = sitk.OtsuMultipleThresholdsImageFilter()
        Otsu.SetNumberOfThresholds(2)
        
        if len(np.unique(FixedArray)) > 2:
            Fixed_Bin = Otsu.Execute(Fixed)
            FixedArray = sitk.GetArrayFromImage(Fixed_Bin)
            FixedArray = (FixedArray == 2) * 1
        
        if len(np.unique(MovingArray)) > 2:
            Moving_Bin = Otsu.Execute(Moving)
            MovingArray = sitk.GetArrayFromImage(Moving_Bin)
            MovingArray = (MovingArray == 2) * 1

    FixedArray = self.Normalize(FixedArray.astype(float))
    MovingArray = self.Normalize(MovingArray.astype(float))


    if Fixed.GetDimension() == 3:
        Array = np.zeros((Fixed.GetSize()[2], Fixed.GetSize()[1], Fixed.GetSize()[0], 3), 'uint8')
        Array[:,:,:,0] = FixedArray
        Array[:,:,:,1] = MovingArray
        Array[:,:,:,2] = MovingArray
        
        if Axis == 'Z':
            if Slice:
                Array = Array[Slice,:,:]
            else:
                Array = Array[Array.shape[0]//2,:,:]
        if Axis == 'Y':
            if Slice:
                Array = Array[:,Slice,:]
            else:
                Array = Array[:,Array.shape[1]//2,:]
        if Axis == 'X':
            if Slice:
                Array = Array[:,:,Slice]
            else:
                Array = Array[:,:,Array.shape[2]//2]

    else:
        Array = np.zeros((Fixed.GetSize()[1], Fixed.GetSize()[0], 3), 'uint8')
        Array[:,:,0] = FixedArray
        Array[:,:,1] = MovingArray
        Array[:,:,2] = MovingArray

    Figure, Axis = plt.subplots()
    Axis.imshow(Array,interpolation=None)
    Axis.axis('Off')
    
    if (Title):
        Axis.set_title(Title)

    if (self.FName):
        plt.savefig(self.FName, bbox_inches='tight', pad_inches=0)

    if self.ShowPlot:
        plt.show()
    else:
        plt.close()

    return

def Intensity(self, Structure, Deformations, Mask=None, Slice=None, Axis='Z', Title=None):

    Array = sitk.GetArrayFromImage(Structure)
    Values = sitk.GetArrayFromImage(Deformations)

    if Mask:
        MaskArray = sitk.GetArrayFromImage(Mask).astype('bool')

    if Structure.GetDimension() == 3:
        
        if Axis == 'Z':
            if Slice:
                Array = Array[Slice,:,:]
                Values = Values[Slice,:,:]
                if Mask:
                    MaskArray = MaskArray[Slice,:,:]
            else:
                Array = Array[Array.shape[0]//2,:,:]
                Values = Values[Values.shape[0]//2,:,:]
                if Mask:
                    MaskArray = MaskArray[MaskArray.shape[0]//2,:,:]

        if Axis == 'Y':
            if Slice:
                Array = Array[:,Slice,:]
                Values = Values[:,Slice,:]
                if Mask:
                    MaskArray = MaskArray[:,Slice,:]
            else:
                Array = Array[:,Array.shape[1]//2,:]
                Values = Values[:,Values.shape[1]//2,:]
                if Mask:
                    MaskArray = MaskArray[:,MaskArray.shape[1]//2,:]

        if Axis == 'X':
            if Slice:
                Array = Array[:,:,Slice]
                Values = Values[:,:,Slice]
                if Mask:
                    MaskArray = MaskArray[:,:,Slice]
            else:
                Array = Array[:,:,Array.shape[2]//2]
                Values = Values[:,:,Values.shape[2]//2]
                if Mask:
                    MaskArray = MaskArray[:,:,MaskArray.shape[2]//2]

    Structure = np.zeros((Array.shape[0], Array.shape[1], 4))
    Structure[:,:,3] = self.Normalize(Array.astype(float)) / 255
    
    if Mask:
        Values[~MaskArray] = np.nan
    else:
        Values[Values == 0] = np.nan

    Figure, Axis = plt.subplots(1,1)
    Plot = Axis.imshow(Values, cmap='jet', vmin=self.IRange[0], vmax=self.IRange[1], interpolation=None)
    Axis.imshow(Structure)
    Axis.axis('Off')

    # Colorbar hack
    Divider = make_axes_locatable(Axis)
    CAxis = Divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(Plot, cax=CAxis, orientation='vertical')

    if (Title):
        Axis.set_title(Title)

    if (self.FName):
        plt.savefig(self.FName, bbox_inches='tight', pad_inches=0)

    if self.ShowPlot:
        plt.show()
    else:
        plt.close()

    return

def Signal(self, X, Y=[], Points=[], Normalize=False, Axes=[], Labels=[], Legend=True):

    if len(X) > 6:
        N = len(X)
        Values = np.ones((N, 4))
        Values[:N//3*2, 0] = np.linspace(1, 0, N//3*2)
        Values[N//3*2:, 0] = 0

        Values[:N//3*2, 2] = np.linspace(0, 1, N//3*2)
        Values[-N//3:, 2] = np.linspace(1, 0, N//3+1)

        Values[:-N//3, 1] = 0
        Values[-N//3:, 1] = np.linspace(0, 1, N//3+1)
        Colors = ListedColormap(Values)(np.linspace(0,1,N))
    else:
        Colors = [(1,0,0), (0,0,1), (0,0,0), (0,1,0), (0,1,1), (1,0,1)]

    Figure, Axis = plt.subplots(1,1)

    if len(Y) == 0:
        self.Y = []
        for ix, x in enumerate(X):
            self.Y.append(x)
            X[ix] = np.arange(len(x))
        Y = self.Y
        delattr(self, 'Y')

    for i in range(len(X)):

        if Normalize:
            Yi = self.Normalize(Y[i])
        else:
            Yi = Y[i]
        Xi = X[i]
        
        if len(Labels) > 0:
            Axis.plot(Xi, Yi, color=Colors[i], label=Labels[i])
        else:
            Axis.plot(Xi, Yi, color=Colors[i], label='Signal ' + str(i+1))

        if len(Points) > 0:
            Px, Py = Xi[Points[i]], Yi[Points[i]]
            Axis.plot(Px, Py, marker='o', color=(0, 0, 0), fillstyle='none', linestyle='none')

    if len(Axes) > 0:
        Axis.set_xlabel(Axes[0])
        Axis.set_ylabel(Axes[1])

    if i < 3:
        Cols = i+1
    else:
        Cols = (1+i)//2

    if Legend:
        plt.legend(loc='upper center', bbox_to_anchor=(0.5,1.12), ncol=Cols)

    if (self.FName):
        plt.savefig(self.FName, bbox_inches='tight', pad_inches=0.02)

    if self.ShowPlot:
        plt.show()
    else:
        plt.close()

    return

def OLS(self, X, Y, Cmap=np.array(None), Labels=None, Alpha=0.95, Annotate=['N','R2','SE','Slope','Intercept']):

    if Labels == None:
        Labels = ['X', 'Y']
    
    # Perform linear regression
    Array = np.array([X,Y])
    if Array.shape[0] == 2:
        Array = Array.T
    Data = pd.DataFrame(Array,columns=['X','Y'])
    FitResults = smf.ols('Y ~ X', data=Data).fit()
    Slope = FitResults.params[1]

    # Build arrays and matrices
    Y_Obs = FitResults.model.endog
    Y_Fit = FitResults.fittedvalues
    N = int(FitResults.nobs)
    C = np.matrix(FitResults.normalized_cov_params)
    X = np.matrix(FitResults.model.exog)

    # Sort X values and Y accordingly
    Sort = np.argsort(np.array(X[:,1]).reshape(len(X)))
    X_Obs = np.sort(np.array(X[:,1]).reshape(len(X)))
    Y_Fit = Y_Fit[Sort]
    Y_Obs = Y_Obs[Sort]

    ## Compute R2 and standard error of the estimate
    E = Y_Obs - Y_Fit
    RSS = np.sum(E ** 2)
    SE = np.sqrt(RSS / FitResults.df_resid)
    TSS = np.sum((FitResults.model.endog - FitResults.model.endog.mean()) ** 2)
    RegSS = TSS - RSS
    R2 = RegSS / TSS
    R2adj = 1 - RSS/TSS * (N-1)/(N-X.shape[1]+1-1)

    ## Compute CI lines
    B_0 = np.sqrt(np.diag(np.abs(X * C * X.T)))
    t_Alpha = t.interval(Alpha, N - X.shape[1] - 1)
    CI_Line_u = Y_Fit + t_Alpha[0] * SE * B_0[Sort]
    CI_Line_o = Y_Fit + t_Alpha[1] * SE * B_0[Sort]

    ## Plots
    DPI = 100
    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=DPI, sharey=True, sharex=True)

    if Cmap.any():
        Colors = plt.cm.winter((Cmap-min(Cmap))/(max(Cmap)-min(Cmap)))
        Scatter = Axes.scatter(X_Obs, Y_Obs, facecolor='none', edgecolor=Colors, marker='o',)
    else:
        Axes.plot(X_Obs, Y_Obs, linestyle='none', marker='o', color=(0,0,1), fillstyle='none')

    Axes.plot(X_Obs, Y_Fit, color=(1,0,0))
    Axes.fill_between(X_Obs, CI_Line_o, CI_Line_u, color=(0, 0, 0), alpha=0.1)

    if Slope > 0:

        YPos = 0.925
        if 'N' in Annotate:
            Axes.annotate(r'$N$  : ' + str(N), xy=(0.025, YPos), xycoords='axes fraction')
            YPos -= 0.075
        if 'R2' in Annotate:
            Axes.annotate(r'$R^2$ : ' + format(round(R2, 2), '.2f'), xy=(0.025, YPos), xycoords='axes fraction')
            YPos -= 0.075
        if 'SE' in Annotate:
            Axes.annotate(r'$SE$ : ' + format(round(SE, 2), '.2f'), xy=(0.025, YPos), xycoords='axes fraction')
        
        YPos = 0.025
        if 'Intercept' in Annotate:
            Intercept = str(FitResults.params[0])
            Round = 3 - Intercept.find('.')
            Intercept = round(FitResults.params[0], Round)
            CI = FitResults.conf_int().loc['Intercept'].round(Round)
            if Round <= 0:
                Intercept = int(Intercept)
                CI = [int(v) for v in CI]
            Text = r'Intercept : ' + str(Intercept) + ' (' + str(CI[0]) + ',' + str(CI[1]) + ')'
            Axes.annotate(Text, xy=(0.425, YPos), xycoords='axes fraction')
            YPos += 0.075

        if 'Slope' in Annotate:
            Round = 3 - str(FitResults.params[1]).find('.')
            Slope = round(FitResults.params[1], Round)
            CI = FitResults.conf_int().loc['X'].round(Round)
            if Round <= 0:
                Slope = int(Slope)
                CI = [int(v) for v in CI]
            Text = r'Slope : ' + str(Slope) + ' (' + str(CI[0]) + ',' + str(CI[1]) + ')'
            Axes.annotate(Text, xy=(0.425, YPos), xycoords='axes fraction')

    elif Slope < 0:

        YPos = 0.025
        if 'N' in Annotate:
            Axes.annotate(r'$N$  : ' + str(N), xy=(0.025, YPos), xycoords='axes fraction')
            YPos += 0.075
        if 'R2' in Annotate:
            Axes.annotate(r'$R^2$ : ' + format(round(R2, 2), '.2f'), xy=(0.025, YPos), xycoords='axes fraction')
            YPos += 0.075
        if 'SE' in Annotate:
            Axes.annotate(r'$SE$ : ' + format(round(SE, 2), '.2f'), xy=(0.025, YPos), xycoords='axes fraction')
        
        YPos = 0.925
        if 'Intercept' in Annotate:
            Intercept = str(FitResults.params[0])
            Round = 3 - Intercept.find('.')
            Intercept = round(FitResults.params[0], Round)
            CI = FitResults.conf_int().loc['Intercept'].round(Round)
            if Round <= 0:
                Intercept = int(Intercept)
                CI = [int(v) for v in CI]
            Text = r'Intercept : ' + str(Intercept) + ' (' + str(CI[0]) + ',' + str(CI[1]) + ')'
            Axes.annotate(Text, xy=(0.425, YPos), xycoords='axes fraction')
            YPos -= 0.075

        if 'Slope' in Annotate:
            Round = 3 - str(FitResults.params[1]).find('.')
            Slope = round(FitResults.params[1], Round)
            CI = FitResults.conf_int().loc['X'].round(Round)
            if Round <= 0:
                Slope = int(Slope)
                CI = [int(v) for v in CI]
            Text = r'Slope : ' + str(Slope) + ' (' + str(CI[0]) + ',' + str(CI[1]) + ')'
            Axes.annotate(Text, xy=(0.425, YPos), xycoords='axes fraction')
    
    Axes.set_xlabel(Labels[0])
    Axes.set_ylabel(Labels[1])
    plt.subplots_adjust(left=0.15, bottom=0.15)

    if (self.FName):
        plt.savefig(self.FName, bbox_inches='tight', pad_inches=0.02)
    if self.ShowPlot:
        plt.show()
    else:
        plt.close()

    return FitResults

def BoxPlot(self, ArraysList, Labels=['', 'Y'], SetsLabels=None, Vertical=True):

    Figure, Axis = plt.subplots(1,1)

    for i, Array in enumerate(ArraysList):
        RandPos = np.random.normal(i,0.02,len(Array))

        Axis.boxplot(Array, vert=Vertical, widths=0.35,
                    showmeans=False,meanline=True,
                    showfliers=False, positions=[i],
                    capprops=dict(color=(0,0,0)),
                    boxprops=dict(color=(0,0,0)),
                    whiskerprops=dict(color=(0,0,0),linestyle='--'),
                    medianprops=dict(color=(0,0,1)),
                    meanprops=dict(color=(0,1,0)))
        Axis.plot(RandPos - RandPos.mean() + i, Array, linestyle='none',
                    marker='o',fillstyle='none', color=(1,0,0))
    
    Axis.plot([],linestyle='none',marker='o',fillstyle='none', color=(1,0,0), label='Data')
    Axis.plot([],color=(0,0,1), label='Median')
    Axis.set_xlabel(Labels[0])
    Axis.set_ylabel(Labels[1])

    if SetsLabels:
        Axis.set_xticks(np.arange(len(SetsLabels)))
        Axis.set_xticklabels(SetsLabels, rotation=0)
    else:
        Axis.set_xticks([])
    
    plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.125))
    plt.subplots_adjust(left=0.25, right=0.75)
    
    if (self.FName):
        plt.savefig(self.FName, bbox_inches='tight', pad_inches=0.02)
    if self.ShowPlot:
        plt.show()
    else:
        plt.close()

def Fabric(self, eValues, eVectors, nPoints=32, Title=None, Angles=[30, 45]):

    # New coordinate system
    Q = np.array(eVectors)

    ## Build data for fabric plotting
    u = np.arange(0, 2 * np.pi + 2 * np.pi / nPoints, 2 * np.pi / nPoints)
    v = np.arange(0, np.pi + np.pi / nPoints, np.pi / nPoints)
    X = eValues[0] * np.outer(np.cos(u), np.sin(v))
    Y = eValues[1] * np.outer(np.sin(u), np.sin(v))
    Z = eValues[2] * np.outer(np.ones_like(u), np.cos(v))
    nNorm = np.zeros(X.shape)

    for i in range(len(X)):
        for j in range(len(X)):
            [X[i, j], Y[i, j], Z[i, j]] = np.dot([X[i, j], Y[i, j], Z[i, j]], Q)
            n = np.array([X[i, j], Y[i, j], Z[i, j]])
            nNorm[i, j] = np.linalg.norm(n)

    NormedColor = nNorm - nNorm.min()
    NormedColor = NormedColor / NormedColor.max()

    Figure = plt.figure(figsize=(5.5, 4))
    Axis = Figure.add_subplot(111, projection='3d')
    Axis.plot_surface(X, Y, Z, facecolors=plt.cm.jet(NormedColor), rstride=1, cstride=1, alpha=0.2, shade=False)
    Axis.plot_wireframe(X, Y, Z, color='k', rstride=1, cstride=1, linewidth=0.1)
    
    # scaling hack
    Bbox_min = np.min([X, Y, Z])
    Bbox_max = np.max([X, Y, Z])
    Axis.auto_scale_xyz([Bbox_min, Bbox_max], [Bbox_min, Bbox_max], [Bbox_min, Bbox_max])
    
    # make the panes transparent
    Axis.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    Axis.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    Axis.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    # make the grid lines transparent
    Axis.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    Axis.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    Axis.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    
    # modify ticks
    MinX, MaxX = -1, 1
    MinY, MaxY = -1, 1
    MinZ, MaxZ = -1, 1
    Axis.set_xticks([MinX, 0, MaxX])
    Axis.set_yticks([MinY, 0, MaxY])
    Axis.set_zticks([MinZ, 0, MaxZ])
    Axis.xaxis.set_ticklabels([MinX, 0, MaxX])
    Axis.yaxis.set_ticklabels([MinY, 0, MaxY])
    Axis.zaxis.set_ticklabels([MinZ, 0, MaxZ])

    Axis.set_xlabel('X')
    Axis.set_ylabel('Y')
    Axis.set_zlabel('Z')

    if (Title):
        Axis.set_title(Title)

    ColorMap = plt.cm.ScalarMappable(cmap=plt.cm.jet)
    ColorMap.set_array(nNorm)
    if not NormedColor.max() == 1:
        ColorBar = plt.colorbar(ColorMap, 
                                ticks=[int(NormedColor.mean() - 1),
                                int(NormedColor.mean()),
                                int(NormedColor.mean() + 1)])
        plt.cm.ScalarMappable.set_clim(self=ColorMap,
                                        vmin=int(NormedColor.mean() - 1),
                                        vmax=int(NormedColor.mean() + 1))
    ColorBar = plt.colorbar(ColorMap)
    ColorBar.set_label('Vector norm (-)')

    # Rotate plot
    Axis.view_init(elev=Angles[0], azim=Angles[1])

    plt.show()

def Stiffness(self, S4, NPoints=32, Angles=[30, 45]):

    I = np.eye(3)

    ## Build data for plotting tensor
    u = np.arange(0, 2 * np.pi + 2 * np.pi / NPoints, 2 * np.pi / NPoints)
    v = np.arange(0, np.pi + np.pi / NPoints, np.pi / NPoints)
    X = np.outer(np.cos(u), np.sin(v))
    Y = np.outer(np.sin(u), np.sin(v))
    Z = np.outer(np.ones_like(u), np.cos(v))
    Color = np.zeros(X.shape)
    for i in range(len(X)):
        for j in range(len(X)):
            n = np.array([X[i, j], Y[i, j], Z[i, j]])
            N = DyadicProduct(n, n)

            Elongation = FrobeniusProduct(N, Transform(S4, N))
            X[i, j], Y[i, j], Z[i, j] = np.array([X[i, j], Y[i, j], Z[i, j]]) * Elongation

            BulkModulus = FrobeniusProduct(I, Transform(S4, N))
            Color[i, j] = BulkModulus

    MinX, MaxX = int(X.min()), int(X.max())
    MinY, MaxY = int(Y.min()), int(Y.max())
    MinZ, MaxZ = int(Z.min()), int(Z.max())

    if Color.max() - Color.min() > 1:
        NormedColor = Color - Color.min()
        NormedColor = NormedColor / NormedColor.max()
    else:
        NormedColor = np.round(Color / Color.max()) / 2

    ## Plot tensor in image coordinate system
    Figure = plt.figure(figsize=(5.5, 4))
    Axis = Figure.add_subplot(111, projection='3d')
    Axis.plot_surface(X, Y, Z, facecolors=plt.cm.jet(NormedColor), rstride=1, cstride=1, alpha=0.2, shade=False)
    Axis.plot_wireframe(X, Y, Z, color='k', rstride=1, cstride=1, linewidth=0.2)
    # scaling hack
    Bbox_min = np.min([X, Y, Z])
    Bbox_max = np.max([X, Y, Z])
    Axis.auto_scale_xyz([Bbox_min, Bbox_max], [Bbox_min, Bbox_max], [Bbox_min, Bbox_max])
    # make the panes transparent
    Axis.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    Axis.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    Axis.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    Axis.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    Axis.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    Axis.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    # modify ticks
    Axis.set_xticks([MinX, 0, MaxX])
    Axis.set_yticks([MinY, 0, MaxY])
    Axis.set_zticks([MinZ, 0, MaxZ])
    Axis.xaxis.set_ticklabels([MinX, 0, MaxX])
    Axis.yaxis.set_ticklabels([MinY, 0, MaxY])
    Axis.zaxis.set_ticklabels([MinZ, 0, MaxZ])

    Axis.xaxis.set_rotate_label(False)
    Axis.set_xlabel('X (MPa)', rotation=0)
    Axis.yaxis.set_rotate_label(False)
    Axis.set_ylabel('Y (MPa)', rotation=0)
    Axis.zaxis.set_rotate_label(False)
    Axis.set_zlabel('Z (MPa)', rotation=90)

    Axis.set_title('Elasticity tensor')
    Axis.view_init(elev=Angles[0], azim=Angles[1])

    ColorMap = plt.cm.ScalarMappable(cmap=plt.cm.jet)
    ColorMap.set_array(Color)
    if not NormedColor.max() == 1:
        ColorBar = plt.colorbar(ColorMap, ticks=[int(Color.mean() - 1), int(Color.mean()), int(Color.mean() + 1)])
        plt.cm.ScalarMappable.set_clim(self=ColorMap, vmin=int(Color.mean() - 1), vmax=int(Color.mean() + 1))
    else:
        ColorBar = plt.colorbar(ColorMap)
    ColorBar.set_label('Bulk modulus (MPa)')
    plt.show()

    return

def Compliance(self, C4, NPoints=32, Angles=[30, 45]):

    C4 = C4 * 1E3
    I = np.eye(3)

    ## Build data for plotting tensor
    u = np.arange(0, 2 * np.pi + 2 * np.pi / NPoints, 2 * np.pi / NPoints)
    v = np.arange(0, np.pi + np.pi / NPoints, np.pi / NPoints)
    X = np.outer(np.cos(u), np.sin(v))
    Y = np.outer(np.sin(u), np.sin(v))
    Z = np.outer(np.ones_like(u), np.cos(v))
    Color = np.zeros(X.shape)
    for i in range(len(X)):
        for j in range(len(X)):
            n = np.array([X[i, j], Y[i, j], Z[i, j]])
            N = DyadicProduct(n, n)

            Elongation = FrobeniusProduct(N, Transform(C4, N))
            X[i, j], Y[i, j], Z[i, j] = np.array([X[i, j], Y[i, j], Z[i, j]]) * Elongation

            BulkModulus = FrobeniusProduct(I, Transform(C4, N))
            Color[i, j] = BulkModulus

    MinX, MaxX = round(X.min(),3), round(X.max(),3)
    MinY, MaxY = round(Y.min(),3), round(Y.max(),3)
    MinZ, MaxZ = round(Z.min(),3), round(Z.max(),3)

    if Color.max() - Color.min() > 1:
        NormedColor = Color - Color.min()
        NormedColor = NormedColor / NormedColor.max()
    else:
        NormedColor = np.round(Color / Color.max()) / 2

    ## Plot tensor in image coordinate system
    Figure = plt.figure(figsize=(5.5, 4))
    Axis = Figure.add_subplot(111, projection='3d')
    Axis.plot_surface(X, Y, Z, facecolors=plt.cm.jet(NormedColor), rstride=1, cstride=1, alpha=0.2, shade=False)
    Axis.plot_wireframe(X, Y, Z, color='k', rstride=1, cstride=1, linewidth=0.2)
    # scaling hack
    Bbox_min = np.min([X, Y, Z])
    Bbox_max = np.max([X, Y, Z])
    Axis.auto_scale_xyz([Bbox_min, Bbox_max], [Bbox_min, Bbox_max], [Bbox_min, Bbox_max])
    # make the panes transparent
    Axis.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    Axis.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    Axis.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    Axis.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    Axis.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    Axis.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    # modify ticks
    Axis.set_xticks([MinX, 0, MaxX])
    Axis.set_yticks([MinY, 0, MaxY])
    Axis.set_zticks([MinZ, 0, MaxZ])
    Axis.xaxis.set_ticklabels([MinX, 0, MaxX])
    Axis.yaxis.set_ticklabels([MinY, 0, MaxY])
    Axis.zaxis.set_ticklabels([MinZ, 0, MaxZ])

    Axis.xaxis.set_rotate_label(False)
    Axis.set_xlabel('X (kPa)', rotation=0)
    Axis.yaxis.set_rotate_label(False)
    Axis.set_ylabel('Y (kPa)', rotation=0)
    Axis.zaxis.set_rotate_label(False)
    Axis.set_zlabel('Z (kPa)', rotation=90)

    Axis.set_title('Compliance tensor')
    Axis.view_init(elev=Angles[0], azim=Angles[1])

    ColorMap = plt.cm.ScalarMappable(cmap=plt.cm.jet)
    ColorMap.set_array(Color)
    if not NormedColor.max() == 1:
        ColorBar = plt.colorbar(ColorMap, ticks=[int(Color.mean() - 1), int(Color.mean()), int(Color.mean() + 1)])
        plt.cm.ScalarMappable.set_clim(self=ColorMap, vmin=int(Color.mean() - 1), vmax=int(Color.mean() + 1))
    else:
        ColorBar = plt.colorbar(ColorMap)
    ColorBar.set_label('Bulk modulus (kPa)')
    plt.show()

    return

def Histogram(self, Arrays, Labels=[], Density=False, Norm=False):

    if len(Arrays) > 6:
        N = len(Arrays)
        Values = np.ones((N, 4))
        Values[:N//3*2, 0] = np.linspace(1, 0, N//3*2)
        Values[N//3*2:, 0] = 0

        Values[:N//3*2, 2] = np.linspace(0, 1, N//3*2)
        Values[-N//3:, 2] = np.linspace(1, 0, N//3+1)

        Values[:-N//3, 1] = 0
        Values[-N//3:, 1] = np.linspace(0, 1, N//3+1)
        Colors = ListedColormap(Values)(np.linspace(0,1,N))
    else:
        Colors = [(1,0,0), (0,0,1), (0,0,0), (0,1,0), (0,1,1), (1,0,1)]

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=300)
    for i, Array in enumerate(Arrays):

        X = pd.DataFrame(Array)
        SortedValues = np.sort(X.T.values)[0]
        N = len(X)
        X_Bar = X.mean()
        S_X = np.std(X, ddof=1)

        if i == 0:
            Bins = 20
            Div = (X.max()-X.min()) / Bins
        else:
            Bins = int(round((X.max()-X.min()) / Div))

        ## Histogram and density distribution
        Histogram, Edges = np.histogram(X, bins=Bins)
        Width = 0.9 * (Edges[1] - Edges[0])
        Center = (Edges[:-1] + Edges[1:]) / 2
        Axes.bar(Center, Histogram, align='center', width=Width, edgecolor=Colors[i], color=(1, 1, 1, 0))
        
        if Density and N < 1E3:
            KernelEstimator = np.zeros(N)
            NormalIQR = np.sum(np.abs(norm.ppf(np.array([0.25, 0.75]), 0, 1)))
            DataIQR = np.abs(X.quantile(0.75)) - np.abs(X.quantile(0.25))
            KernelHalfWidth = 0.9 * N ** (-1 / 5) * min(np.abs([S_X, DataIQR / NormalIQR]))
            for Value in SortedValues:
                KernelEstimator += norm.pdf(SortedValues - Value, 0, KernelHalfWidth * 2)
            KernelEstimator = KernelEstimator / N
        
            Axes.plot(SortedValues, KernelEstimator, color=Colors[i], label='Kernel Density')
        
        if Norm:
            TheoreticalDistribution = norm.pdf(SortedValues, X_Bar, S_X)
            Axes.plot(SortedValues, TheoreticalDistribution, linestyle='--', color=Colors[i], label='Normal Distribution')
        
    if len(Labels) > 0:
        plt.xlabel(Labels[0])
        plt.ylabel(Labels[1])
    
    # plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.15), prop={'size': 10})
    # plt.legend(loc='upper left')

    if (self.FName):
        plt.savefig(self.FName, bbox_inches='tight', pad_inches=0.02)
    if self.ShowPlot:
        plt.show()
    else:
        plt.close()



#%% Main
# Main code

def Main(File):

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
    Parser.add_argument('File', help='File to process (required)', type=str)

    # Read arguments from the command line
    Arguments = Parser.parse_args()

    Main(Arguments.File)
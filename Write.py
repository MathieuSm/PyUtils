#%% #!/usr/bin/env python3
# Initialization

Version = '01'

Description = """
    Short description of the analysis performed by this script

    Version Control:
        01 - Original script

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel, University of Bern

    Date: Month Year
    """

#%% Imports
# Modules import

import os
import h5py
import pathlib
import argparse
import numpy as np
from strenum import StrEnum
from xml.dom import minidom
from numpy.typing import ArrayLike

from Time import Process, Update


#%% Functions
# Define functions

def __init__(self):
    self.Echo = True
    self.FName = 'Image'
    pass

def VectorFieldVTK(self, VectorField, SubSampling=1):

    if self.Echo:
        Text = 'Write VTK'
        Process(1, Text)

    # Determine 2D of 3D vector field
    Dimension = VectorField.shape[-1]
    Size = VectorField.shape[:-1]

    if Dimension == 2:

        Size = np.append(1,Size)

        # Build Grid point
        z, y, x = np.arange(0, Size[0], SubSampling), np.arange(0, Size[1], SubSampling), np.arange(0, Size[2], SubSampling)
        NumberOfElements = len(x) * len(y) * len(z)

        # Build vector arrays
        u = VectorField[:, :, 0]
        v = VectorField[:, :, 1]
        w = np.zeros(NumberOfElements).reshape(Size[1:])

    elif Dimension == 3:

        # Build Grid point
        z = np.arange(0, Size[0], SubSampling)
        y = np.arange(0, Size[1], SubSampling)
        x = np.arange(0, Size[2], SubSampling)
        NumberOfElements = len(x) * len(y) * len(z)

        # Build vector arrays
        u = VectorField[:, :, :, 0]
        v = VectorField[:, :, :, 1]
        w = VectorField[:, :, :, 2]

    File = open(self.FName + '.vtk','w')

    # ASCII file header
    File.write('# vtk DataFile Version 4.2\n')
    File.write('VTK from Python\n')
    File.write('ASCII\n\n')
    File.write('DATASET RECTILINEAR_GRID\n')
    File.write('DIMENSIONS ' + str(len(x)) + ' ' + str(len(y)) + ' ' + str(len(z)) + '\n\n')
    File.close()

    # Append ascii x,y,z
    MaxLineWidth = Size[2]*Size[1]*Size[0]
    File = open(self.FName + '.vtk','a')
    File.write('X_COORDINATES ' + str(len(x)) + ' int\n')
    File.write(np.array2string(x.astype('int'),
                            max_line_width=MaxLineWidth,
                            threshold=MaxLineWidth)[1:-1] + '\n')
    File.write('\nY_COORDINATES ' + str(len(y)) + ' int\n')
    File.write(np.array2string(y.astype('int'),
                            max_line_width=MaxLineWidth,
                            threshold=MaxLineWidth)[1:-1] + '\n')
    File.write('\nZ_COORDINATES ' + str(len(z)) + ' int\n')
    File.write(np.array2string(z.astype('int'),
                            max_line_width=MaxLineWidth,
                            threshold=MaxLineWidth)[1:-1] + '\n\n')
    File.close()

    # Append another subheader
    File = open(self.FName + '.vtk','a')
    File.write('\nPOINT_DATA ' + str(NumberOfElements) + '\n\n')
    File.write('VECTORS Deformation float\n')
    File.close()

    # Append ascii u,v,w and build deformation magnitude array
    Magnitude = np.zeros(VectorField.shape[:-1])
    File = open(self.FName + '.vtk','a')

    if Dimension == 2:
        for j in range(0,Size[1],SubSampling):
            if self.Echo:
                Update(j / Size[1], 'Write Vectors')
            for i in range(0,Size[2],SubSampling):
                Magnitude[j,i] = np.sqrt(u[j,i]**2 + v[j,i]**2 + w[j,i]**2)
                File.write(str(u[j,i]) + ' ' + str(v[j,i]) + ' ' + str(w[j,i]) + ' ')

    elif Dimension == 3:
        for k in range(0,Size[0],SubSampling):
            if self.Echo:
                Update(k / Size[0], 'Write Vectors')
            for j in range(0,Size[1],SubSampling):
                for i in range(0,Size[2],SubSampling):
                    Magnitude[k,j,i] = np.sqrt(u[k,j,i] ** 2 + v[k,j,i] ** 2 + w[k,j,i] ** 2)
                    File.write(str(u[k,j,i]) + ' ' + str(v[k,j,i]) + ' ' + str(w[k,j,i]) + ' ')

    File.close()

    # Append another subheader
    File = open(self.FName + '.vtk', 'a')
    File.write('\n\nSCALARS Magnitude float\n')
    File.write('LOOKUP_TABLE default\n')

    if Dimension == 2:
        for j in range(0, Size[1], SubSampling):
            if self.Echo:
                Update(j / Size[1], 'Write Magnitudes')
            for i in range(0, Size[2], SubSampling):
                File.write(str(Magnitude[j,i]) + ' ')

    elif Dimension == 3:
        for k in range(0, Size[0], SubSampling):
            if self.Echo:
                Update(k / Size[0], 'Write Magnitudes')
            for j in range(0, Size[1], SubSampling):
                for i in range(0, Size[2], SubSampling):
                    File.write(str(Magnitude[k,j,i]) + ' ')

    File.close()

    if self.Echo:
        Process(0,Text)

    return

def FabricVTK(self, eValues, eVectors, nPoints=32, Scale=1, Origin=(0,0,0)):

    if self.Echo:
        Text = 'Write Fabric VTK'
        Process(1, Text)

    ## New coordinate system
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

    # Scale the arrays
    X = Scale/2 * X
    Y = Scale/2 * Y
    Z = Scale/2 * Z

    # Translate origin to the center of ROI
    X = Origin[2] + X + Scale/2
    Y = Origin[1] + Y + Scale/2
    Z = Origin[0] + Z + Scale/2

    # Write VTK file
    VTKFile = open(self.FName + '.vtk', 'w')

    # Write header
    VTKFile.write('# vtk DataFile Version 4.2\n')
    VTKFile.write('VTK from Python\n')
    VTKFile.write('ASCII\n')
    VTKFile.write('DATASET UNSTRUCTURED_GRID\n')

    # Write points coordinates
    Points = int(X.shape[0] * X.shape[1])
    VTKFile.write('\nPOINTS ' + str(Points) + ' floats\n')
    for i in range(Points):
        if self.Echo:
            Update(i / len(Points), 'Write Points')
        VTKFile.write(str(X.reshape(Points)[i].round(3)))
        VTKFile.write(' ')
        VTKFile.write(str(Y.reshape(Points)[i].round(3)))
        VTKFile.write(' ')
        VTKFile.write(str(Z.reshape(Points)[i].round(3)))
        VTKFile.write('\n')

    # Write cells connectivity
    Cells = int(nPoints**2)
    ListSize = int(Cells*5)
    VTKFile.write('\nCELLS ' + str(Cells) + ' ' + str(ListSize) + '\n')

    ## Add connectivity of each cell
    Connectivity = np.array([0, 1])
    Connectivity = np.append(Connectivity,[nPoints+2,nPoints+1])

    for i in range(Cells):
        if self.Echo:
            Update(i / len(Cells), 'Write Cells')

        VTKFile.write('4')

        if i > 0 and np.mod(i,nPoints) == 0:
            Connectivity = Connectivity + 1

        for j in Connectivity:
            VTKFile.write(' ' + str(j))
        VTKFile.write('\n')

        ## Update connectivity
        Connectivity = Connectivity+1

    # Write cell types
    VTKFile.write('\nCELL_TYPES ' + str(Cells) + '\n')
    for i in range(Cells):
        VTKFile.write('9\n')

    # Write MIL values
    VTKFile.write('\nPOINT_DATA ' + str(Points) + '\n')
    VTKFile.write('SCALARS MIL float\n')
    VTKFile.write('LOOKUP_TABLE default\n')

    for i in range(nPoints+1):
        if self.Echo:
            Update(i / len(nPoints+1), 'Write Values')
        for j in range(nPoints+1):
            VTKFile.write(str(nNorm.reshape(Points)[j + i * (nPoints+1)].round(3)))
            VTKFile.write(' ')
        VTKFile.write('\n')
    VTKFile.close()

    if self.Echo:
        Process(0, Text)

    return

def XDMF(self):

    """
    Write XDMF files for visualization in Paraview
    Adapted from https://github.com/tdegeus/XDMFWrite_h5py
    """
    
    
    class ElementType(StrEnum):
        """
        Element types:

        -   Polyvertex
        -   Triangle
        -   Quadrilateral
        -   Hexahedron
        """

        Polyvertex = "Polyvertex"
        Triangle = "Triangle"
        Quadrilateral = "Quadrilateral"
        Hexahedron = "Hexahedron"


    class AttributeCenter(StrEnum):
        """
        Attribute centers:

        -   Cell
        -   Node
        """

        Cell = "Cell"
        Node = "Node"


    class Field:
        """
        Base class of XDMF-fields.

        :param dataset: HDF5-dataset.
        :param name: Name to use in the XDMF-file [default: same as dataset].
        """

        def __init__(self, dataset: h5py.File, name: str):

            self.filename = dataset.parent.file.filename
            self.path = dataset.name
            self.shape = dataset.shape
            self.shape_str = " ".join(str(i) for i in self.shape)
            self.name = name

            if self.name is None:
                self.name = dataset.name

        def __iter__(self):
            return iter(self.__list__())

        def relpath(self, path: str) -> str:
            """
            Change the path of the HDF5-file to a path relative to another file (the XDMF-file).
            :param path: Path to make the file relative to.
            """
            self.filename = os.path.relpath(self.filename, pathlib.Path(path).parent)

        def __str__(self) -> str:
            """
            Return XML snippet.
            """
            return minidom.parseString("\n".join(self.__list__())).toprettyxml(newl="")


    class Geometry(Field):
        """
        Interpret a dataset as a Geometry (aka nodal-coordinates / vertices).

        :param dataset: The dataset.
        """

        def __init__(self, dataset: h5py.Group):
            super().__init__(dataset, "Geometry")
            assert len(self.shape) == 2

        def __list__(self) -> list[str]:
            """
            :return: XDMF code snippet.
            """

            ret = []

            if self.shape[1] == 1:
                ret += ['<Geometry GeometryType="X">']
            elif self.shape[1] == 2:
                ret += ['<Geometry GeometryType="XY">']
            elif self.shape[1] == 3:
                ret += ['<Geometry GeometryType="XYZ">']
            else:
                raise OSError("Illegal number of dimensions.")

            ret += [
                (
                    f'<DataItem Dimensions="{self.shape_str}" Format="HDF"> '
                    f"{self.filename}:{self.path} </DataItem>"
                )
            ]
            ret += ["</Geometry>"]

            return ret


    class Topology(Field):
        """
        Interpret a dataset as a Topology (aka connectivity).

        :param dataset: Dataset.
        :param element_type: Element-type (see :py:class:`ElementType`).
        """

        def __init__(self, dataset: h5py.Group, element_type: ElementType):
            super().__init__(dataset, "Topology")
            self.element_type = element_type

            if not shape_is_correct(self.shape, self.element_type):
                raise OSError("Incorrect dimensions for type")

        def __list__(self) -> list[str]:
            """
            :return: XDMF code snippet.
            """

            ret = []
            ret += [
                f'<Topology NumberOfElements="{self.shape[0]:d}" TopologyType="{self.element_type}">'
            ]
            ret += [
                (
                    f'<DataItem Dimensions="{self.shape_str}" Format="HDF"> '
                    f"{self.filename}:{self.path} </DataItem>"
                )
            ]
            ret += ["</Topology>"]

            return ret


    class Attribute(Field):
        """
        Interpret a dataset as an Attribute.

        :param dataset: Dataset.
        :param center: How to center the Attribute (see :py:class:`AttributeCenter`).
        :param name: Name to use in the XDMF-file [default: same as dataset]
        """

        def __init__(self, dataset: h5py.File, center: str, name: str = None):
            super().__init__(dataset, name)
            self.center = center
            assert len(self.shape) > 0
            assert len(self.shape) < 3

        def __list__(self) -> list[str]:
            """
            :return: XDMF code snippet.
            """

            if len(self.shape) == 1:
                t = "Scalar"
            elif len(self.shape) == 2:
                t = "Vector"
            else:
                raise OSError("Type of data cannot be deduced")

            ret = []
            ret += [f'<Attribute AttributeType="{t}" Center="{self.center}" Name="{self.name}">']
            ret += [
                (
                    f'<DataItem Dimensions="{self.shape_str}" Format="HDF"> '
                    f"{self.filename}:{self.path} </DataItem>"
                )
            ]
            ret += ["</Attribute>"]

            return ret


    class File:
        """
        Base class of XDMF-files.
        The class allows (requires) to open the file in context-manager mode.

        :param filename: Filename of the XDMF-file.
        :param mode: Write mode.
        """

        def __init__(self, filename: str, mode: str = "w"):
            self.filename = filename
            self.mode = mode
            self.lines = []

        def __iter__(self):
            return iter(self.__list__())

        def __str__(self) -> str:
            return minidom.parseString("\n".join(self.__list__())).toprettyxml(newl="")

        def __list__(self) -> list[str]:
            return _asfile(self.lines)

        def __add__(self, content: Field):
            """
            Add content to file.
            :param content: Content to add.
            """

            if isinstance(content, list):
                self.lines += content
                return self

            if isinstance(content, Field):
                content.relpath(self.filename)  # todo: operation that does not modify "content"
                self.lines += list(content)
                return self

            self.lines += [content]
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            with open(self.filename, self.mode) as file:
                file.write(str(self))


    class TimeStep:
        """
        Mark a time-step in a :py:class:`TimeSeries`.

        :param name: Name of the time step.
        :param time: Value of time
        """

        def __init__(self, name: str = None, time: float = None):
            self.name = name
            self.time = time


    class Grid(File):
        """
        XDMF-file with one grid. The grid can contain:

        -   :py:class:`Geometry`.
        -   :py:class:`Topology`.
        -   :py:class:`Attribute`.
        -   :py:class:`Structured`.
        -   :py:class:`Unstructured`.

        See :py:class:`Structured` or :py:class:`Unstructured` for suggested usage.

        :param filename: Filename of the XDMF-file.
        :param mode: Write mode.
        :param name: Name of the grid.
        """

        def __init__(self, filename: str, mode: str = "w", name: str = "Grid"):
            super().__init__(filename, mode)
            self.name = name

        def __list__(self) -> list[str]:

            ret = []
            ret += [f'<Grid CollectionType="Temporal" GridType="Collection" Name="{self.name}">']
            ret += [f'<Grid Name="{self.name}">']
            ret += self.lines
            ret += ["</Grid>"]
            ret += ["</Grid>"]

            return _asfile(ret)


    class TimeSeries(File):
        r"""
        XDMF-file with a series of 'time-steps' of grids, separated by :py:class:`TimeStep`.
        The grid can contain:

        -   :py:class:`Geometry`.
        -   :py:class:`Topology`.
        -   :py:class:`Attribute`.
        -   :py:class:`Structured`.
        -   :py:class:`Unstructured`.

        Usage::

            with h5py.File("my.h5", "w") as file, xh.TimeSeries("my.xdmf") as xdmf:

                file["coor"] = coor
                file["conn"] = conn

                for i in range(4):

                    file[f"/stress/{i:d}"] = float(i) * stress
                    file[f"/disp/{i:d}"] = float(i) * xh.as3d(disp)

                    xdmf += xh.TimeStep()
                    xdmf += xh.Unstructured(file["coor"], file["conn"], xh.ElementType.Quadrilateral)
                    xdmf += xh.Attribute(file[f"/disp/{i:d}"], xh.AttributeCenter.Node, name="Disp")
                    xdmf += xh.Attribute(file[f"/stress/{i:d}"], xh.AttributeCenter.Cell, name="Stress")

        :param name: Name of the TimeSeries.
        """

        def __init__(self, filename: str, mode: str = "w", name: str = "TimeSeries"):
            super().__init__(filename, mode)
            self.name = name
            self.start = []
            self.settings = []

        def __add__(self, other: TimeStep):

            if isinstance(other, TimeStep):
                self.start += [len(self.lines)]
                self.settings += [other]
                return self

            super().__add__(other)
            return self

        def __list__(self) -> list[str]:

            ret = []
            ret += [f'<Grid CollectionType="Temporal" GridType="Collection" Name="{self.name}">']

            start = [i for i in self.start] + [len(self.lines)]

            for i in range(len(self.start)):

                if self.settings[i].name is None:
                    name = f"Increment {i:d}"
                else:
                    name = self.settings[i].name

                if self.settings[i].time is None:
                    t = i
                else:
                    t = self.settings[i].time

                ret += [f'<Grid Name="{name}">']
                ret += [f'<Time Value="{str(t)}"/>']
                ret += self.lines[start[i] : start[i + 1]]  # noqa: E203
                ret += ["</Grid>"]

            ret += ["</Grid>"]
            return _asfile(ret)


    class _Grid(Field):
        """
        Base class for a grid.
        """

        def __init__(
            self, dataset_geometry: h5py.Group, dataset_topology: h5py.Group, element_type: ElementType
        ):
            self.geometry = Geometry(dataset_geometry)
            self.topology = Topology(dataset_topology, element_type)

        def __iter__(self):
            return iter(self.__list__())

        def relpath(self, path: str):
            self.geometry.relpath(path)
            self.topology.relpath(path)

        def __list__(self) -> list[str]:
            return list(self.geometry) + list(self.topology)


    class Structured(_Grid):
        """
        Interpret DataSets as a Structured (individual points).
        Short for the concatenation of:

        -   ``Geometry(file["coor"])``
        -   ``Topology(file["conn"], ElementType.Polyvertex)``.

        Usage::

            with h5py.File("my.h5", "w") as file, xh.Grid("my.xdmf") as xdmf:

                file["coor"] = coor
                file["conn"] = conn
                file["radius"] = radius

                xdmf += xh.Structured(file["coor"], file["conn"])
                xdmf += xh.Attribute(file["radius"], "Node")

        :param dataset_geometry: Geometry dataset.
        :param dataset_topology: Mock Topology ``numpy.arange(N)``, ``N`` = number of nodes (vertices).
        """

        def __init__(self, dataset_geometry: h5py.Group, dataset_topology: h5py.Group):
            super().__init__(dataset_geometry, dataset_topology, ElementType.Polyvertex)


    class Unstructured(_Grid):
        """
        Interpret DataSets as a Unstructured
        (Geometry and Topology, aka nodal-coordinates and connectivity).
        Short for the concatenation of:

        -   ``Geometry(file["coor"])``
        -   ``Topology(file["conn"], element_type)``.

        Usage::

            with h5py.File("my.h5", "w") as file, xh.Unstructured("my.xdmf") as xdmf:

                file["coor"] = coor
                file["conn"] = conn
                file["stress"] = stress

                xdmf += xh.Unstructured(file["coor"], file["conn"], "Quadrilateral")
                xdmf += xh.Attribute(file["stress"], "Cell")

        :param dataset_geometry: Path to the Geometry dataset.
        :param dataset_topology: Path to the Topology dataset.
        :param element_type: Element-type (see :py:class:`ElementType`).
        """

        def __init__(
            self, dataset_geometry: h5py.Group, dataset_topology: h5py.Group, element_type: ElementType
        ):
            super().__init__(dataset_geometry, dataset_topology, element_type)

    #%% Functions
    # Define some functions

    def shape_is_correct(shape: ArrayLike, element_type: ElementType) -> bool:
        """
        Check that a shape matches the expected shape for a certain type.

        :param shape: Shape of a dataset.
        :param element_type: Element-type (see :py:class:`ElementType`).
        :return: `True` is the shape is as expected (no guarantee that the data is correct).
        """

        if len(shape) == 1 and element_type == ElementType.Polyvertex:
            return True

        if len(shape) != 2:
            return False

        if shape[1] == 3 and element_type == ElementType.Triangle:
            return True

        if shape[1] == 4 and element_type == ElementType.Quadrilateral:
            return True

        if shape[1] == 8 and element_type == ElementType.Hexahedron:
            return True

        return False


    def as3d(arg: ArrayLike) -> ArrayLike:
        r"""
        Return a list of vectors as a list of vectors in 3d (as required by ParaView).

        :param [N, d] arg: Input array (``d <= 3``).
        :return: The array zero-padded such that the shape is ``[N, 3]``
        """

        assert arg.ndim == 2

        if arg.shape[1] == 3:
            return arg

        ret = np.zeros([arg.shape[0], 3], dtype=arg.dtype)
        ret[:, : arg.shape[1]] = arg
        return ret


    def _asfile(lines: list[str]) -> str:
        """
        Convert a list of lines to an XDMF-file.
        :param lines: List of lines.
        :return: XDMF-file.
        """
        ret = []
        ret += ['<Xdmf Version="3.0">']
        ret += ["<Domain>"]
        ret += lines
        ret += ["</Domain>"]
        ret += ["</Xdmf>"]
        return ret


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
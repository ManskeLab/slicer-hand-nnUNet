"""
Parameter Node wrapper class for handCBCT

Could be merged into Logic.py, but would not make much sense semantically
"""

from typing import Annotated

from slicer import vtkMRMLScalarVolumeNode, vtkMRMLSegmentationNode
from slicer.parameterNodeWrapper import parameterNodeWrapper, Choice


@parameterNodeWrapper
class handCBCTParameterNode:
    """
    Parameters for handCBCT module

    inputVolume - volume to segment
    foldCount - number of folds for nnUNet model configuration
    deviceType - the device type for nnUNet model configuration
    outputSegment - segmentation result
    """

    inputVolume: vtkMRMLScalarVolumeNode
    foldCount: Annotated[int, Choice([1, 2, 3, 4, 5])] = 1 # default to 1 for performance purposes
    deviceType: Annotated[str, Choice(["cuda", "cpu", "mps"])] = "cuda"
    outputSegment: vtkMRMLSegmentationNode


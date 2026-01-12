import logging

import slicer
import slicer.util
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic

#
# handCBCTLogic
#

class handCBCTLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        self.segmentationLogic = None
        self.dependenciesInstalled = False
        self.is_setup = False

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        
        """ Code provided by Extension Wizard
        if not parameterNode.GetParameter("Threshold"):
            parameterNode.SetParameter("Threshold", "100.0")
        if not parameterNode.GetParameter("Invert"):
            parameterNode.SetParameter("Invert", "false")
        """
    def process(self, inputVolume, outputVolume):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be segmented (nii expected)
        :param outputVolume: segmentation result

        """
        if not self.is_setup:
          self.setup()

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time
        startTime = time.time()
        logging.info('Processing started')
        
        # TODO: enable modification of parameters, specifically the fold count
        self.segmentationLogic.startSegmentation(inputVolume)
        
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')


        

    def installDependencies(self):
      """
      Install dependencies utilizing the SlicerNNuNet extension
      """
      try:
          import SlicerNNUNetLib
      except ModuleNotFoundError as err:
          slicer.util.errorDisplay("This module requires the SlicerNNUNet extension. Please install it in Extension Manager.")
          raise err

      from SlicerNNUNetLib import InstallLogic
      install_logic = InstallLogic()
      install_logic.progressInfo.connect(print) # TODO: review later whether we wish to log somewhere else
      install_logic.setupPythonRequirements()

      self.dependenciesInstalled = True

    def setup(self):
      """
      Setup logic including installing dependencies, loading model weight, and defining self.segmentationLogic
      """
      if not self.dependenciesInstalled:
        self.installDependencies()

      # SlicerNNUNetLib is installed
      from SlicerNNUNetLib import SegmentationLogic

      self.segmentationLogic = SegmentationLogic()
      self.segmentationLogic.progressInfo.connect(print)
      self.segmentationLogic.errorOccurred.connect(slicer.util.errorDisplay)
      self.segmentationLogic.inferencedFinished.connect(self.segmentationLogic.loadSegmentation)
      
      self.loadWeights()
      self.is_setup = True

    def loadWeights(self):
      """
      Load weights for nnUNet from folder
      Folder specifications: https://github.com/KitwareMedical/SlicerNNUnet?tab=readme-ov-file#expected-weight-folder-structure

      See the SlicerNNUNet Parameter class for more details
      """
      if not self.dependenciesInstalled:
        self.installDependencies()

      from SlicerNNUNetLib import Parameter, SegmentationLogic
      self.modelParameters = Parameter(modelPath = self.getModelPath())
      if self.hasValidParams:
        slicer.util.messageBox("Model directory is valid.")
      else:
        slicer.util.messageBox("Model directory is not valid.")


      self.segmentationLogic.setParameter(self.modelParameters)


    def getModelPath(self):
      """
      Get path to model directory, download to here
      """
      from pathlib import Path
      path = Path(__file__).parent
      return path.joinpath("..", "Resources", "Model").resolve()
    
    @property
    def hasValidParams(self):
      if self.modelParameters:
        return self.modelParameters.isValid()
      else:
        return False
      
    

      



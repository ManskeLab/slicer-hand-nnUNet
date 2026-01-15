import logging
from pathlib import Path
import slicer
import slicer.util
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic

from .Parameter import handCBCTParameterNode

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


    module_dir = slicer.util.modulePath("handCBCT")

    def __init__(self):
      """
      Called when the logic class is instantiated. Can be used for initializing member variables.
      """
      ScriptedLoadableModuleLogic.__init__(self)
      self.segmentationLogic = None
      self.modelParameters = None

      # flags for setup related tasks
      self.dependenciesInstalled = False
      self.is_setup = False
    
    def getParameterNode(self):
      """
        Return a Parameter Node 
      """
      return handCBCTParameterNode(super().getParameterNode())
    
    def process(self, inputVolume, outputSegment):
      """
      Run the processing algorithm.
      Can be used without GUI widget.
      :param inputVolume: volume to be segmented (nii expected)
      :param outputVolume: segmentation result

      """
      if not self.is_setup:
        self.setup()

      if not inputVolume or not outputSegment:
        raise ValueError("Input or output selected is invalid")

      import time
      startTime = time.time()
      logging.info('Processing started')
      
      # TODO: enable modification of parameters, specifically the fold count. Integrate with parameter nodes, or provide an update function if run from GUI.
      self.segmentationLogic.startSegmentation(inputVolume)
      
      
      stopTime = time.time()
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

      # connect Segmentation signals
      self.segmentationLogic.progressInfo.connect(print)
      self.segmentationLogic.errorOccurred.connect(slicer.util.errorDisplay)
      self.segmentationLogic.inferenceFinished.connect(self.segmentationLogic.loadSegmentation)
      
      # finish set up 
      self.loadWeights() # TODO: add download method for weights if not present
      self.is_setup = True

    def loadWeights(self):
      """
      Load weights for nnUNet from folder
      Folder specifications: https://github.com/KitwareMedical/SlicerNNUnet?tab=readme-ov-file#expected-weight-folder-structure

      See the SlicerNNUNetLib Parameter class for more details
      """
      if not self.dependenciesInstalled:
        self.installDependencies()

      from SlicerNNUNetLib import Parameter, SegmentationLogic

      # get model path and create directory if it does not exist
      modelPath = self.getModelPath()
      modelPath.mkdir(parents = True, exists_ok = True)


      self.modelParameters = Parameter(modelPath = modelPath)

      # testing purposes, check whether the directory is valid
      if self.hasValidParams:
        slicer.util.messageBox("Model directory is valid.")
      else:
        slicer.util.messageBox("Model directory is not valid.")

      # attach loaded model parameters to segmentation logic
      self.segmentationLogic.setParameter(self.modelParameters)


    def downloadWeights(self) -> bool:
      """
      Download weights for nnUNet model, present on github.
      
      :return: boolean indicating success
      :rtype: bool
      """

      pass
    @staticmethod
    def getModelPath(self) -> Path:
      """
      Path to model directory.

      Module should download weights folder to here.
      Provide this directory to SlicerNNUNetLib for module loading.
      """
      
      return self.getCachePath() / "Model"
    
    @staticmethod
    def getCachePath() -> Path:
      """
      Path to cache directory for this module, use to store downloaded model weight
      
      :return: path to the module's cache directory
      :rtype: pathlib.Path
      """
      return Path(slicer.app.cachePath) / "handCBCT"

    @property
    def hasValidParams(self) -> bool:
      """
      Validity of current model parameters as loaded from self.modelPath
      """
      if self.modelParameters:
        modelResponse = self.modelParameters.isValid()
        print(modelResponse[1])
        return modelResponse[0]
      else:
        return False
      
    

      



import logging
from pathlib import Path

import slicer
import slicer.util
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic
from slicer import vtkMRMLScalarVolumeNode, vtkMRMLSegmentationNode

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

    # model constants
    MODEL_CHECKPOINT = "checkpoint_final.pth"
    MODEL_WEIGHT_NAME = "Dataset001_hand"

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
    
    def process(self, inputVolume, foldCount, deviceType, outputSegment):
      """
      Run the processing algorithm.
      Can be used without GUI widget.

      :param inputVolume: volume to be segmented (.nii expected)
      :param foldCount: number of folds for nnunet
      :param deviceType: device type used 
      :param outputVolume: segmentation result

      See handCBCTParameterNode for more details
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
      self.segmentationLogic.waitForSegmentationFinished()
      
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
      
      # prepare nnunet Parameter
      from SlicerNNUNetLib import Parameter
      self.modelParameters = Parameter()
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


      # get model path and create directory if it does not exist
      modelPath = self.getModelPath()
      modelPath.mkdir(parents = True, exists_ok = True)


      self.modelParameters.modelPath = str(modelPath)
      self.modelParameters.checkPointName = handCBCTLogic.MODEL_CHECKPOINT

      # testing purposes, check whether the directory is valid
      if self.hasValidParams:
        slicer.util.messageBox("Model directory is valid.")
      else:
        slicer.util.messageBox("Model directory is not valid.")

      # attach updated model parameters to segmentation logic
      self._reloadParameters()


    def downloadWeights(self, downloadAgain: bool = False) -> bool:
      """
      Download weights for nnUNet model, present on github.

      :param downloadAgain: boolean switch to force download even if file already exists
      :type downloadAgain: bool
      :return: boolean indicating success of download
      :rtype: bool
      """
      
      # obtain release URL (adapted from https://github.com/gaudot/SlicerDentalSegmentator/blob/main/DentalSegmentator/DentalSegmentatorLib/PythonDependencyChecker.py)

      from github import Github, GithubException
      gh = Github()
      repo = gh.get_repo("ManskeLab/slicer-hand-nnUNet")
      assets = [asset for release in repo.get_releases() for asset in release.get_assets() if asset.name == handCBCTLogic.MODEL_WEIGHT_NAME + ".zip"]

      url = assets[0].browser_download_url
      
      weightPath = self.getModelPath() / handCBCTLogic.MODEL_WEIGHT_NAME

      if not weightPath.exists() or downloadAgain:

        if downloadAgain:
          import shutil
          shutil.rmtree(weightPath)

        slicer.util.messageBox("Downloading model. This may take some time.")


        import requests
        session = requests.Session()
        response = session.get(url, stream = True)
        
        response.raise_for_status()

        zipPath = str(weightPath) + ".zip"
        with open(zipPath, "wb") as f:
          for chunk in response.iter_content(1024 * 1024):
            f.write(chunk)


        import zipfile
        
        with zipfile.ZipFile(zipPath, "r") as f:
            f.extractall(weightPath)

        return True
      else:
        return False
        

    
    def _reloadParameters(self) -> None:
      """
      Reattach parameters to self.segmentationLogic

      Call this when reconfiguring for parameter node values
      """
      if self.segmentationLogic and self.modelParameters:
        self.segmentationLogic.setParameter(self.modelParameters)


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

      :return: boolean value representing whether the current modelParameters are linked to a valid nnunet weights directory
      :rtype: bool
      """
      if self.modelParameters:
        modelResponse = self.modelParameters.isValid()
        print(modelResponse[1])
        return modelResponse[0]
      else:
        return False
      
    @property
    def weightsExist(self) -> bool:
      """
      Docstring for weightsExist
      
      :return: boolean value representing whether weights directory exists (has been downloaded)
      :rtype: bool
      """

      return (self.getModelPath() / handCBCTLogic.MODEL_WEIGHT_NAME).exists()

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

    def __init__(self, log_method):
      """
      Called when the logic class is instantiated. Can be used for initializing member variables.
      
      :param log_method: provide a method for logging output
      """
      ScriptedLoadableModuleLogic.__init__(self)
      
      # attributes
      self.log_method = log_method # TODO: change current messages which create UI elements (ie. MessageBox) to use log_method
      self.segmentationLogic = None
      self.modelParameters = None
      self.segmentResult = None

      # flags for setup related tasks
      self.dependenciesInstalled = False
      self.is_setup = False
    
    def getParameterNode(self):
      """
        Return a Parameter Node 
      """
      return handCBCTParameterNode(super().getParameterNode())
    
    def process(self, inputVolume: vtkMRMLScalarVolumeNode, foldCount: int, deviceType: str, outputSegment: vtkMRMLSegmentationNode):
      """
      Run the processing algorithm.
      Can be used without GUI widget.

      :param inputVolume: volume to be segmented (.nii expected)
      :param foldCount: number of folds for nnunet
      :param deviceType: device type used 
      :param outputSegment: segmentation result

      See handCBCTParameterNode for more details
      """
      # check whether model has been configured
      if not self.is_setup:
        self.setup()
    
      if not self.hasValidParams:
        raise RuntimeError("Invalid Model! Try downloading again.")
    
    
      # check valid inputs
      if not inputVolume or not outputSegment:
        raise ValueError("Input or output selected is invalid")

      self.inputName = inputVolume.GetName() # unused
      self.segmentResult = outputSegment
      
      # update parameters
      self.modelParameters.folds = handCBCTLogic.produceFoldString(foldCount)
      self.modelParameters.device = deviceType
      self._reloadParameters()
      
      logging.info('Processing started')

      # begin segmentation
      self.segmentationLogic.startSegmentation(inputVolume)
      
    def stopProcess(self):
      """
      Stops the segmentation process
      """
      self.segmentationLogic.stopSegmentation()
      
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
      try:
        install_logic.setupPythonRequirements()
      except Exception as e:
        slicer.util.errorDisplay("Error occurred while downloading requirements.")
        raise e
      
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
      self.segmentationLogic.inferenceFinished.connect(self._inferenceFinished) 
      # TODO: reconfigure signal to connect to custom method; currently experiencing issues with loadSegmentation [fixed]
      
      # prepare nnunet Parameter
      from SlicerNNUNetLib import Parameter
      self.modelParameters = Parameter()

      if not (self.getModelPath() / handCBCTLogic.MODEL_WEIGHT_NAME).exists():
        self.downloadWeights()

      self.loadWeights() # loadWeights will download weights if not already downloaded
      self.is_setup = True

    def loadWeights(self):
      """
      Load weights for nnUNet from folder
      Folder specifications: https://github.com/KitwareMedical/SlicerNNUnet?tab=readme-ov-file#expected-weight-folder-structure

      See the SlicerNNUNetLib Parameter class for more details
      """
      if not self.dependenciesInstalled:
        self.installDependencies()

      
      # get model path and check if it exists, download if it does not exist
      modelPath = self.getModelPath() / handCBCTLogic.MODEL_WEIGHT_NAME
      if not modelPath.exists():
        # avoid tying loading with download
        # self.downloadWeights()
        slicer.util.messageBox("Model directory does not exist.")
        return

      if not self.modelParameters:
        from SlicerNNUNetLib import Parameter
        self.modelParameters = Parameter()
      
      self.modelParameters.modelPath = modelPath
      self.modelParameters.checkPointName = handCBCTLogic.MODEL_CHECKPOINT

      # testing purposes, check whether the directory is valid
      if self.hasValidParams:
        slicer.util.messageBox("Model directory is valid.")
      else:
        slicer.util.messageBox("Model directory is not valid.")

      # attach updated model parameters to segmentation logic
      self._reloadParameters()


    def downloadWeights(self, downloadAgain: bool = False, progressBar = None) -> bool:
      """
      Download weights for nnUNet model, present on github.

      :param downloadAgain: boolean switch to force download even if file already exists
      :param progressBar: optionally provide a slier progressBar to update
      :type downloadAgain: bool
      :return: boolean indicating success of download
      :rtype: bool
      """
      
      # obtain release URL (code adapted from https://github.com/gaudot/SlicerDentalSegmentator/blob/main/DentalSegmentator/DentalSegmentatorLib/PythonDependencyChecker.py)

      from github import Github, GithubException
      gh = Github()
      repo = gh.get_repo("ManskeLab/slicer-hand-nnUNet")
      assets = [asset for release in repo.get_releases() for asset in release.get_assets() if asset.name == handCBCTLogic.MODEL_WEIGHT_NAME + ".zip"]

      url = assets[0].browser_download_url
      
      weightPath = self.getModelPath() / handCBCTLogic.MODEL_WEIGHT_NAME


      # check whether to download 
      if not weightPath.exists() or downloadAgain:

        if downloadAgain and weightPath.exists():
          import shutil
          shutil.rmtree(weightPath)

        # message output
        if progressBar:
          progressBar.setLabelText("Downloading model. This may take some time.")
        else:
          slicer.util.messageBox("Downloading model. This may take some time.")

        weightPath.mkdir(parents = True)

        import requests
        session = requests.Session()
        response = session.get(url, stream = True)
        
        response.raise_for_status()

        zipPath = str(weightPath) + ".zip"

        # set maximum 
        if progressBar:
          progressBar.maximum = int(assets[0].size)

        
        # write to zipPath in chunks
        with open(zipPath, "wb") as f:
          downloaded = 0
          for chunk in response.iter_content(1024 * 1024):
            downloaded += len(chunk)
            if progressBar:
                progressBar.value = downloaded
                slicer.app.processEvents()
            f.write(chunk)


        import zipfile
        
        with zipfile.ZipFile(zipPath, "r") as f:
            f.extractall(weightPath)

        slicer.util.messageBox("Download complete.")
        return True
      else:
        slicer.util.messageBox("Already downloaded.")
        return False
        


    def _inferenceFinished(self, *args, **kwargs):
      """
      Wraps segmentationLogic.inferenceFinished
      
      Load in returned segmentation from loadSegmentation to segmentResult
      """
      
      if not self.segmentResult or not self.segmentResult.IsA("vtkMRMLSegmentationNode"):
        slicer.util.errorDisplay("No destination segmentation node set.")
        return
      
      result: vtkMRMLSegmentationNode = self.segmentationLogic.loadSegmentation()
      
      if not result:
        slicer.util.errorDisplay("Inference finshed, but not segmentation was generated.")
      
      destination_segment = self.segmentResult.GetSegmentation()
      destination_segment.DeepCopy(result.GetSegmentation())
      
      # TODO: deal with the temporary loaded segmentation
      
      slicer.util.messageBox("Inference complete.")
      

    def _reloadParameters(self) -> None:
      """
      Reattach parameters to self.segmentationLogic

      Call this when reconfiguring for parameter node values
      """
      if self.segmentationLogic and self.modelParameters:
        self.segmentationLogic.setParameter(self.modelParameters)


    @staticmethod
    def getModelPath() -> Path:
      """
      Path to model directory.

      Module should download weights folder to here.
      Provide this directory to SlicerNNUNetLib for module loading.
      """
      
      return handCBCTLogic.getCachePath() / "Model"
    
    @staticmethod
    def getCachePath() -> Path:
      """
      Path to cache directory for this module, use to store downloaded model weight
      
      :return: path to the module's cache directory
      :rtype: pathlib.Path
      """
      return Path(slicer.app.cachePath) / "handCBCT"


    @staticmethod
    def produceFoldString(folds: int) -> str:
      """
      Docstring for produceFoldString
      
      :param folds: number of folds to prepare string for
      :return: string for input into SlicerNNuNet Parameter class representing fold count
      :rtype: str
      """
      
      return ",".join(str(i) for i in range(folds))

    @property
    def hasValidParams(self) -> bool:
      """
      Validity of current model parameters as loaded from self.modelPath

      :return: boolean value representing whether the current modelParameters are linked to a valid nnunet weights directory
      :rtype: bool
      """
      if self.modelParameters:
        modelResponse = self.modelParameters.isValid()
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

# slicer-hand-nnUNet
3D Slicer module implementing nnUNet-model for hand segmentation


## Usage Instructions 

1. **Requirements:** Install requirements from Slicer's Extension Manager
- Slicer NNuNet extension

2. **Install:** Install the handCBCT module

3. **Loading:** Load CBCT scan into Slicer

4. **Module:** Open module in Modules -> Segmentations -> handCBCT

4. **Setup:** Select appropriate input, output, fold count (suggested: 1), and device type

5. **Segmentation:** Press start to begin segmentation. Check status in Slicer's integrated Python Console.
- If issues occur, check troubleshooting for more information

6. **Results** If desired, use the "remove small islands" button to post process, removing undesired segment islands
- Be aware that all disconnected islands of segments will be removed.
- You may want to use the built-in Segment Editor module for additional edits
- Recommended: Use the 'Erase', 'Paint', and 'Fill between slices tools' for manual segment editing if needed
- Exporting can be done in Slicer's Segmentations module.


## Troubleshooting
If the segmentation gets stuck at 100% and "done with volume" has been outputted, press the Stop button to manually end the process and load the results

Intel MacOS Requirements:
- numpy <2
- torch <2.3 (with nnunetv2 2.5.2)
- See: https://github.com/MIC-DKFZ/nnUNet/issues/2742


## Credit
https://github.com/KitwareMedical/SlicerNNUnet

https://github.com/gaudot/SlicerDentalSegmentator


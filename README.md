# Medical-Image-Annotator
Annotating landmarks (e.g. embryonic cell centroids) in medical grayscale images with graphical image annotation tool.

## Getting Started
This is a modification of the image annotation tool [labelImg](https://github.com/tzutalin/labelImg). All modifications were performed using Python 3.6 and PyQt5. 

### Installation
Installation instructions for PyQt5 and required libraries are avalaible [here](https://github.com/tzutalin/labelImg#build-from-source).

## Usage
### Annotation GUI
The GUI is initiated by running:
```
python labelImg.py
```
Shortcuts can be found [here](https://github.com/tzutalin/labelImg#hotkeys). Note that this version creates landmarks, not rectangular boxes. As such, only the x,y landmark coordinates and number of objects are saved in the XML file.

<img src="https://github.com/llockhar/medical-image-annotator/blob/master/demoImages/AnnotatorGUI.png" alt="Annotator GUI" width="600" />

### Saving Outputs as 2D Mask Array
It may be desirable to save landmark coordinates as a 2D mask array for further analysis. This is done by initializing a 2D array of 0s, then assigning 1 to centroid coordinates. Additionally, Gaussian smoothing can be applied to balance foreground/background distribution.
Gaussian smoothed landmark masks are created by running.
```
python xmltomask.py --xml_path path_to_xml_file --mask_path path_to_desired_output \
--resize_dim  optional_height_width_int_for_resizing --sigma optional_gaussian_kernel_sigma
```
To overlay the mask on original images, add the following arguments:
```
python xmltomask.py --xml_path path_to_xml_fils --mask_path path_to_desired_output \
--resize_dim optional_height_width_int_for_resizing --sigma optional_gaussian_kernel_sigma \
--overlay True --img_path path_to_original_images --overlay_path path_to_desired_overlay_output
```

<img src="https://github.com/llockhar/medical-image-annotator/blob/master/demoImages/EmbryoImage.png" alt="Embryo Image" height="300" /><img src="https://github.com/llockhar/medical-image-annotator/blob/master/demoImages/CentroidMask.png" alt="Embryo Centroid Mask" height="300" /><img src="https://github.com/llockhar/medical-image-annotator/blob/master/demoImages/MaskOverlay.png" alt="Embryo Mask Overlay" height="300" />



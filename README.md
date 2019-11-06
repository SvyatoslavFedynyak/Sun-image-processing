# Sun-image-processing
Automatic detection and segmentation of different solar features taken by PICARD/SODISM  Satellite  (dark regions (Sunspots), bright  or white regions,   active region (networks and faculae) using: 
* Machine learning
* Region growing technique.
* Thresholding 
  
Modified region growing technique to detect the regions of interest (RoI)
FAR error rate and FRR error rate  for active regions
The detection performance is enhanced further using a combination of modified region growing and neural network (NN) technique which is trained on statistical features extracted from the RoI and non-RoI. Using this combination the FAR has dropped to 2% for active regions, and 4% for filaments.

# Usefull links
Scikit-image module for image segmentation
https://towardsdatascience.com/image-segmentation-using-pythons-scikit-image-module-533a61ecc980
About FAR & FRR
https://www.bayometric.com/false-acceptance-rate-far-false-recognition-rate-frr/

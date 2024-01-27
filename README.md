# Implement Deep Neural network on DeepCEST
INTRODUCTION:
In this work, I learn how to achieve predict Lorentzian parameters 5-pool CEST MRI spectra with
corresponding uncertainty maps from uncorrected raw MRI Z-spectra using Deep neural networks as well
as reconstruct the prediction image.
As per research paper, the input data for a neural feed-forward network consisted of 7 T in vivo uncorrected
Z-spectra of a single B1 level, and a B1 map. The 7 T raw data were acquired using a 3D snapshot gradient
echo multiple interleaved mode saturation CEST sequence. These inputs were mapped voxel-wise to target
data consisting of Lorentzian amplitudes generated conventionally by 5-pool Lorentzian fitting of
normalized, denoised, B0- and B1-corrected Z-spectra. The deepCEST network was trained with Mean
square Error or Gaussian negative log-likelihood loss, providing an uncertainty quantification in addition
to the Lorentzian amplitudes Z-spectra of a single B1 level, and a B1 map.

METHODS
A Deep neural network used for processing pipeline to perform reconstruction of CEST contrasts image.
The conventional pipeline with the proposed deepCEST 7 T scheme uses in vivo Z-spectra and evaluated by
conventional methods using the evaluation pipeline described in Figure 1. Also, it was shown that neural
networks (NN) can be used and are effective to automate and accelerate the reconstruction from an
uncorrected CEST spectrum, forming the deepCEST approach. This deep NN is included into
the online image reconstruction of the scanner system to predict the CEST contrasts in ∼30 s with
uncertainty quantification to indicate the trustworthiness of the predictions.

![image](https://github.com/abulzunayed/Model_DeepCEST/assets/122612945/46539848-4109-4dd0-aa92-95e72ae75005)

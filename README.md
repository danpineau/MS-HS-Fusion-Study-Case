# MS-HS Fusion with Spectrally Varying Blurs : Study Case with SOTA methods

## 1. Download data and libraries

Data can be downloaded here :

https://filesender.renater.fr/?s=download&token=bdeb8c0d-3b7e-422b-876b-df8ef47e90e4

and the files need to be placed in a "data" folder.

Multiple libraries are required for this project, and among them is aljabr, which can be found here : https://github.com/forieux/aljabr


## 2. Change paths

Some absolute paths need to be changed in the following files :
- run_fusion_methods.py
- fusion_tools.py
- reco_quality_metrics.py
- criteria_definitions.py


## 3. Run the script "run_fusion_methods.py"

Three methods are implemented here :
- l2 regularization with iterative minimization [1]
- l2 regularization with explicit minimization [2]
- l2/l1 with Majorize-Minimize algorithm [3]

The expected outputs with these methods have already been saved in the "output" folder.

## Contact

For any issues regarding the implementation, contact me at dan.pineau@uni.lu

## References

[1] C. Guilloteau, T. Oberlin, O. Berné and N. Dobigeon, "Hyperspectral and Multispectral Image Fusion Under Spectrally Varying Spatial Blurs – Application to High Dimensional Infrared Astronomical Imaging," in IEEE Transactions on Computational Imaging, vol. 6, pp. 1362-1374, 2020, doi: 10.1109/TCI.2020.3022825.

[2] D. Pineau, F. Orieux and A. Abergel, "Exact Solution for Multispectral and Hyperspectral Fusion Via Hessian Inversion," 2023 13th Workshop on Hyperspectral Imaging and Signal Processing: Evolution in Remote Sensing (WHISPERS), Athens, Greece, 2023, pp. 1-5, doi: 10.1109/WHISPERS61460.2023.10431252, 

[3] D. Pineau, F. Orieux and A. Abergel, "Multispectral and Hyperspectral Image Fusion with Spectrally Varying Blurs and MM Algorithm," in IEEE Transactions on Computational Imaging, doi: 10.1109/TCI.2025.3565138.



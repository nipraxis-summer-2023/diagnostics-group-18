# Analysis Draft Plan

## Pre-processing
 - Motion correction 
 - Skull stripping (?)
 - Spatial Normalization (TBC)
 - Intensity Normalization (TBC)


 ## Outlier detection
 - Average volume intensity comparison (average from full scanning sequence)
 - DVARS (absolute difference in volume intensity between subsequent volumes)
 - Account for intensity changes due to simulus using correlation between stimuli and voxels?
 - Calculate volumes which are not in the IQR?

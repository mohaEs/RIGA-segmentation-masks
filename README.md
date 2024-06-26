# RIGA segmentation masks
 
## Downloand the masks 
This repository contains the soft and hard segmentation masks of the fundus images of RIGA dataset. (*-Masks directories) </br>
The segmentation masks of each annotatators are also provided. (*-segments directories) </br>
You can simply download the the zip file: https://github.com/mohaEs/RIGA-segmentation-masks/raw/main/RIGA_masks.zip

Extracted masks: </br>
<img src="./images/img1.png" width="850" title="input"> </br>

original dataset annotations: </br>
<img src="./images/img2.png" width="500" title="input">


## Citation 

If you used this dataset, please cite one of the following reports: </br>
- https://iovs.arvojournals.org/article.aspx?articleid=2791240 
Eslami, M., Motati, L.S., Kalahasty, R., Hashemabad, S.K., Shi, M., Luo, Y., Tian, Y., Zebardast, N., Wang, M. and Elze, T., 2023. Deep Learning based Adversarial Disturbances in Fundus Image Analysis. Investigative Ophthalmology & Visual Science, 64(9), pp.PB002-PB002. </br>
- https://iovs.arvojournals.org/article.aspx?articleid=2791086 
Motati, L.S., Kalahasty, R., Hashemabad, S.K., Shi, M., Luo, Y., Tian, Y., Zebardast, N., Wang, M., Elze, T. and Eslami, M., 2023. Evaluation of Robustness of Disc/Cup Segmentation in Different Fundus Imaging Conditions. Investigative Ophthalmology & Visual Science, 64(8), pp.1129-1129.</br> 
- https://iovs.arvojournals.org/article.aspx?articleid=2791055 
Kalahasty, R., Motati, L.S., Hashemabad, S.K., Shi, M., Luo, Y., Tian, Y., Zebardast, N., Wang, M., Elze, T. and Eslami, M., 2023. Evaluation of Landmark Localization Models for Fundus Imaging Conditions. Investigative Ophthalmology & Visual Science, 64(8), pp.267-267.

And of course, you need to cite the original RIGA dataset as well: </br>
- https://doi.org/10.7302/Z23R0R29 </br>
- https://doi.org/10.1117/12.2293584

## Reproduction

There is no need to reproduce the masks and segments, but if you are interested you can try the following procedure: 

- requirements </br>
use the requirements file for creating your environment. If you use the pip version, make sure to install opencv separately.

- download the RIGA dataset and unzip the same as the following tree: </br>
https://deepblue.lib.umich.edu/data/concern/data_sets/3b591905z

```
./
./BinRushed
./BinRushed\BinRushed1-Corrected
./BinRushed\BinRushed2
./BinRushed\BinRushed3
./BinRushed\BinRushed4
./Magrabia
./Magrabia\MagrabiaMale
./Magrabia\MagrabiFemale
./MESSIDOR
```

- extract masks of individual images

for each subset, there is a main file:

> python main_Messidor.py </br>
> python main_Margrebia.py </br>
> python main_BinRushed.py </br>

Notice that, since the BinRushed subset contains jpeg images, the process for extracting the maps is more complicated and will take much more time.

- combine all masks </br>
In this step, all the segments of each prime are combined, and soft and hard masks are generated. 

> python main_CombineMasks.py

The untitled files are used for developing and also manual fixation of the ones that were skipped in the above automated process.

## Sources:

The region growing part is modified from: </br>
https://github.com/Spinkoo/Region-Growing

# Official Implementation of SeqHand
This is the implementation of synthetic hand motion generation presented in [SeqHAND (ECCV 2020)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570120.pdf).

## required libraries
```
pytorch==1.3.1
cv2
imageio
pickle
os
pillow
numpy
opendr>=0.76
```

## BH-dataset
I have normalized [BigHand](http://bjornstenger.github.io/papers/yuan_cvpr2017.pdf) dataset annotations (3D joint coordinates for each pose data).
You need to get this [normalized BigHand data file](https://drive.google.com/file/d/13iiZDkxA3hCR6l4L4Em2Dxo6jBTvkBLM/view?usp=sharing) and save at 
your project directory in order to generate hand motions.

## MANO
Install and set up your [manopth (MANO library)](https://github.com/hassony2/manopth) as explained in their work page.
Put your "mano" folder in /your/project/directory/mano/

## Hand Skin Color
Hand skin colors (meshes_colored/) are obtained from our [baseline work](https://github.com/boukhayma/3dhand)

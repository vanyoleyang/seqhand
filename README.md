# Official Implementation of SeqHand
This is the implementation of synthetic hand motion generation presented in [SeqHAND (ECCV 2020)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570120.pdf).

## required libraries
..coming sonn..

## BH-dataset
I have normalized [BigHand](http://bjornstenger.github.io/papers/yuan_cvpr2017.pdf) dataset annotations (3D joint coordinates for each pose data) and re-ordered them in MANO order.
You need to get this [normalized BigHand file](https://drive.google.com/file/d/13iiZDkxA3hCR6l4L4Em2Dxo6jBTvkBLM/view?usp=sharing) and save at 
your project directory in order to generate hand motions.

## MANO
[MANO library](https://github.com/hassony2/manopth)
Put your mano folder from MANO lib in /your/project/directory/mano/

## Hand Skin Color
Hand skin colors are obtained from our [baseline work](https://github.com/boukhayma/3dhand)

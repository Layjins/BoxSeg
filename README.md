## Quick start


### Preliminaries

BoxSeg is based on [detectron2](https://github.com/facebookresearch/detectron2), [Adelaidet](https://github.com/aim-uofa/AdelaiDet), and BoxTeacher (https://github.com/hustvl/BoxTeacher).


1. Install dependencies for BoxSeg.

```bash
# install detectron2
python setup.py build develop

# install adelaidet
cd AdelaiDet
python setup.py build develop
cd ..
```

2. Prepare the datasets for BoxSeg.

```
BoxSeg
datasets/
 - coco/
 - voc/
 - cityscapes/
```
You can refer to [detectron-doc](datasets/README.md) for more details about (custom) datasets.

3. Prepare the pre-trained weights for different backbones.

```bash
mkdir pretrained_models
cd pretrained_models
# download the weights with the links from the above table.
```

### Training

```bash
python train_net.py --config-file <path/to/config> --num-gpus 8
```

### Testing

```bash
python train_net.py --config-file <path/to/config> --num-gpus 8 --eval MODEL.WEIGHTS <path/to/weights>
```




## Acknowledgements

BoxSeg is based on [detectron2](https://github.com/facebookresearch/detectron2), [Adelaidet](https://github.com/aim-uofa/AdelaiDet), and BoxTeacher (https://github.com/hustvl/BoxTeacher).
We sincerely thanks for their code and contribution to the community!


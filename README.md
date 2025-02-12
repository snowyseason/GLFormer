# GLFormer(Under review)

[Class-Aware Transformer Coupling Local Context for Weakly Supervised Object Localization in Optical Remote Sensing Images]().

## Environment Setup
- Ubuntu 20.04, with Python 3.8 and the following python dependencies.
```
pip install -r requirements.txt

## Data Preparation 
<details>
<summary>
C45V2
</summary>

- Download [the C45V2 development kit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012).
  ``` bash
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  tar –xvf VOCtrainval_11-May-2012.tar
  ```
- Download augmented annoations `SegmentationClassAug.zip` from [SBD dataset](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6126343&casa_token=cOQGLW2KWqUAAAAA:Z-QHpQPf8Pnb07A75yBm2muYjqJwYUYPFbwwxMFHRcjRX0zl45kEGNqyTEPH7irB2QbabZbn&tag=1) via this [link](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0).
- Make your data directory like this below
  ``` bash
  VOCdevkit/
  └── VOC2012
      ├── Annotations
      ├── ImageSets
      ├── JPEGImages
      ├── SegmentationClass
      ├── SegmentationClassAug
      └── SegmentationObject
    ```

  </details>

  <details>
  <summary>
  MS COCO 2014
  </summary>
  
  - Download [MS COCO 2014 dataset](https://cocodataset.org/#home)
    ``` bash
    wget http://images.cocodataset.org/zips/train2014.zip
    wget http://images.cocodataset.org/zips/val2014.zip
    ```
    </details>

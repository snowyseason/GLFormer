# Class-Aware Transformer Coupling Local Context for Weakly Supervised Object Localization in Optical Remote Sensing Images (Under Review)

**[Official PyTorch implementation of Class-Aware Transformer Coupling Local Context for Weakly Supervised Object Localization in Optical Remote Sensing Images]()**

## Environment Setup
- **OS**: Ubuntu 20.04
- **Python**: 3.8
- **Dependencies**:
  ```bash
  pip install -r requirements.txt

## Data Preparation

The project expects the following directory structure. Please organize your files accordingly:

```bash
├── datasets/
│   ├── C45V2/                  # Images for C45V2
│   └── PN2/                    # Images for PN2
└── metadata/
    ├── C45V2/
    │   ├── test/
    │   │   ├── class_labels.txt
    │   │   ├── image_ids.txt
    │   │   ├── image_sizes.txt
    │   │   └── localization.txt
    │   ├── train/
    │   │   └── ... (same files as test)
    │   └── val/
    │       └── ... (same files as test)
    └── PN2/
        ├── test/
        │   ├── class_labels.txt
        │   ├── image_ids.txt
        │   ├── image_sizes.txt
        │   └── localization.txt
        ├── train/
        │   └── ... (same files as test)
        └── val/
            └── ... (same files as test)
```
## Get Start

### Installation
Clone the repository:

```
git clone https://github.com/snowyseason/GLFormer
cd GLFormer
```

### Training
To train the model, simply run:
```
bash train.sh
```

### Evaluation
To evaluate the trained model, run:
```
bash eval.sh
```

## Acknowledgements
Part of our evaluation and training code is based on [wsolevaluation(CVPR2020)](https://github.com/clovaai/wsolevaluation). Thanks for their works and sharing.

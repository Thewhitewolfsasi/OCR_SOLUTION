# Metal Stamping Character Recognition System
### Problem Statement
In small-scale industries, the conventional practice of assigning serial numbers to manufactured parts by physically punching them poses challenges in efficient data retrieval. To overcome this, there is a critical need to leverage Optical Character Recognition (OCR) technology for automating the process of serial number extraction from part images. The existing manual methods are not only time-consuming but also susceptible to errors, especially under adverse environmental conditions.

## Installation

Download the Project as zip file

```bash
  cd OCR_SOLUTION
  pip install -r requirement.txt
```
## Text Detection
- Download the trained models
 
 *Model name* | *Used datasets* | *Languages* | *Purpose* | *Model Link* |
 | :--- | :--- | :--- | :--- | :--- |
General | SynthText, IC13, IC17 | Eng + MLT | For general purpose | [Click](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ)
IC15 | SynthText, IC15 | Eng | For IC15 only | [Click](https://drive.google.com/open?id=1i2R7UIUqmkUtF0jv_3MXTqmQ_9wuAnLf)
LinkRefiner | CTW1500 | - | Used with the General Model | [Click](https://drive.google.com/open?id=1XSaFwBkOaFOdtk4Ane3DFyJGPRw6v5bO)

* Run with pretrained model
``` (with python 3.7)
python test.py --trained_model=[weightfile] --test_folder=[folder path to test images] --char True
```

The result image and score maps will be saved to `./result` by default.

### Arguments
* `--trained_model`: pretrained model
* `--text_threshold`: text confidence threshold
* `--low_text`: text low-bound score
* `--link_threshold`: link confidence threshold
* `--cuda`: use cuda for inference (default:True)
* `--canvas_size`: max image size for inference
* `--mag_ratio`: image magnification ratio
* `--poly`: enable polygon type result
* `--show_time`: show processing time
* `--test_folder`: folder path to input images
* `--refine`: use link refiner for sentense-level dataset
* `--refiner_model`: pretrained refiner model
* `--char`: Character level split (default:False). If True, means Each character is got identified used for Model Deployment

## Text Recognition Dataset
- Download the Datasets
 
 *Used datasets* | *Dataset Link* |
| :--- | :--- |
DATASET | [Click](https://drive.google.com/file/d/1AV2bxOPfvxJhp4bhwQNEg9b8ErX31NjW/view?usp=sharing)
EMNIST | [Click](https://www.nist.gov/itl/products-and-services/emnist-dataset)
NIST | [Click](https://www.nist.gov/srd/nist-special-database-19)
SVHN | [Click](http://ufldl.stanford.edu/housenumbers/)
Custom Dataset | Cannot disclose Internship Dataset

## Text Detection Results
- Bounding Box of the Image

![image](https://github.com/Thewhitewolfsasi/OCR_SOLUTION/assets/127896918/e209eb9a-6e85-4d66-8d0a-7d103bc14297)

- Region Score shows Heatmap of the probability of a pixel being at the center of a character

![image](https://github.com/Thewhitewolfsasi/OCR_SOLUTION/assets/127896918/0bf91fea-9159-46a6-a679-78e2168b62b3)

- Affinity Score shows Heatmap of the probability of the space between adjacent characters

![image](https://github.com/Thewhitewolfsasi/OCR_SOLUTION/assets/127896918/5b2bbe88-6370-4f5f-889b-654b94009d76)

## Text Recognition Results
- Recognised Text shows above the bounding box

<p align="center" width="100%">
    <img width="100%" src="https://github.com/Thewhitewolfsasi/OCR_SOLUTION/assets/127896918/2f82c218-a572-4aa9-939a-ce5f9fa09b1b">
</p>

<p align="center" width="100%">
    <img width="100%" src='https://github.com/Thewhitewolfsasi/OCR_SOLUTION/assets/127896918/2f29781c-861c-4308-ad6e-fcec5b4c8c4d'>
</p>

<p align="center" width="100%">
    <img width="100%" src='https://github.com/Thewhitewolfsasi/OCR_SOLUTION/assets/127896918/b0c21b12-0de0-4446-8d34-e4057732a83a'>
</p>

<p align="center" width="100%">
    <img width="100%" src="https://github.com/Thewhitewolfsasi/OCR_SOLUTION/assets/127896918/27e794a3-2777-4669-aeda-b66c447a3482">
</p>

## Acknowledgements

 - [Craft Text Detection](https://github.com/clovaai/CRAFT-pytorch/tree/master)

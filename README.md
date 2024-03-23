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

The result image and socre maps will be saved to `./result` by default.

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

## Acknowledgements

 - [Craft Text Detection](https://github.com/clovaai/CRAFT-pytorch/tree/master)

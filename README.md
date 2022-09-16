# COCO converter to YOLO OBB

The objective of this repository is to create a YOLO_OBB dataset from existing COCO dataset.

YOLO OBB is a version of YOLO_v5 with oriented bounding boxes : https://github.com/hukaixuan19970627/yolov5_obb

## Program utilisation

#### Installation

Clone the repository
```
git clone https://github.com/tcotte/coco2yolo_obb
```

Install dependencies
```
pip install -r requirement.txt
```

Respect the COCO file hierarchy.

#### COCO file hierarchy

```
COCO dataset
├── img
     |────1.jpg/png
     |────...
     └────10000.jpg/png
├── ann
     |────{json_file}.txt
```

#### YOLO OBB hierarchy

```
YOLO dataset
├── images
     |────1.jpg/png
     |────...
     └────10000.jpg/png
├── labelTxt
     |────1.txt
     |────...
     └────10000.txt
```


#### Class.yaml file

Indicate all the classes into the yaml file. For example :
```
0: cat
1: dog
```


### COCO --> YOLO_OBB

To convert coco dataset type : 
```
python coco2yolo.py --class_file {class.yaml} --output {output_path} --input {input_path}
```

```
optional arguments:
  -h, --help            show this help message and exit
  --class_file CLASS_FILE
                        YAML file path containing classes
  --output OUTPUT       Output YOLO directory
  --input INPUT         Input COCO directory
```

### Verification

Verification of the YOLO_OBB generated dataset can be done typing this command : 

```
python visualisation.py --source {input_img/dir} --output_path {output}
```

```
optional arguments:
  -h, --help            show this help message and exit
  --source SOURCE       Path of the input picture or input directory of images
  --output_path OUTPUT_PATH
                        Output directory of annotated pictures
  --draw_r DRAW_R       Draw the oriented bounding box
  --draw_sa DRAW_SA     Draw the main symmetric axis of bounding box
```

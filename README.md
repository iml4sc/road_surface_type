# Road Surface Type Classification Project

    road_surface_type/
    ├── data/
    │    ├── __init__.py
    │    ├── transform.py
    │    └── custom_folder.py
    ├── models/
    │    ├── __init__.py
    │    └── train.py
    ├── utils/
    │    ├── __init__.py
    │    ├── dir_ops.py
    │    ├── evaluation.py
    │    ├── metrics.py
    │    └── plotting.py
    ├── prediction_scripts/
    │    ├── input/
    │    │    └── [csv_to_json.py]
    │    ├── input_module.py
    │    ├── model_module.py
    │    ├── output_module.py
    │    ├── preprocessing.py
    │    ├── postprocessing.py
    │    └── [predict_main.py]
    ├── [train_main.py]
    ├── [test_main.py]
    └── requirements.txt
    
> pip install -r requirements.txt

>
    python train_main.py \
        --models resnet152v2 densenet121 \
        --data-dir /path/to/dataset(Including Train,Val Folders) \
        --epochs 50 \
        --batch-size 32 \
        --lr 1e-4

>
    python test_main.py \
        --weights ./results/results_resnet152v2_full/resnet152v2_model.pth \
        --data-dir /path/to/dataset(Including Test Folder)

>
    python predict_main.py \
        --geopose-json input/input_geopose.json \
        --model-path ./results/results_resnet152v2_full/resnet152v2_model.pth \
        --output-dir prediction_results_resnet152v2 \
        --device cuda

>
    python prediction_scripts/input/csv_to_json.py --csv-path TEST_all_3type.csv --output-json input_geopose.json

>
## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Acknowledgment
This work was developed in part at the Integrated Media Systems Center (IMSC),
University of Southern California.  
Please acknowledge IMSC-USC in any academic or research use of this code.

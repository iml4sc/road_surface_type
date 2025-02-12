# roadtype_01

    roadtype_01/
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
    ├── train_main.py
    ├── test_main.py
    └── requirements.txt

> pip install -r requirements.txt

>
    python train_main.py \
        --models resnet50 densenet121 \
        --data-dir /path/to/dataset \
        --epochs 50 \
        --batch-size 32 \
        --lr 1e-4
        --use-crop \


>
    python test_main.py \
        --weights ./results/resnet50_crop/resnet50_model.pth \
        --data-dir /path/to/dataset \



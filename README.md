# High quality, fast, modular reference implementation of SSD in PyTorch 1.0

## Develop Guide

If you want to add your custom components, please see [DEVELOP_GUIDE.md](DEVELOP_GUIDE.md) for more details.

1/ Chuyển đổi dữ liệu sang format PASCAL VOC:

+ Clone về source trên Google Colab:

```text
!git clone https://github.com/ontheskyl/SSD.git
%cd SSD
```

+ Chạy file data_preprocess.py:

File này sẽ phân chia dữ liệu theo 3 tập (train, validation, test) với validation mặc định 0.2

python convert_pascalvoc_dataset/data_preprocess.py <data_direction> <output_annotation_path> <test_ratio>

data_direction: đường dẫn dữ liệu ban đầu

output_annotation_path: đường dẫn kết quả trả về (chương trình sẽ trả về 2 file Train_annotation.txt và Test_annotation.txt

test_ratio: tỉ lệ dữ liệu testing

Ví dụ: 
```text
!python convert_pascalvoc_dataset/data_preprocess.py "/content/drive/MyDrive/Colab Notebooks/Sunshine Tech/cmnd_back" "/content/drive/MyDrive/Colab Notebooks/Sunshine Tech/Annotations" 0.1
```

+ Chạy file build.py:

File này sẽ dựa vào 3 file Train_Annotations.txt, Val_Annotations.txt, Test_Annotations.txt để xây dựng format PASCAL VOC

python convert_pascalvoc_dataset/data_preprocess.py <annotation_path> <output_direction>

output_direction: đường dẫn kết quả trả về (chương trình sẽ tạo dữ liệu theo chuẩn PASCAL VOC)

annotation_path: đường dẫn đến 2 file Train_annotation.txt và Test_annotation.txt đã tạo trước đó

Ví dụ:
```text
!python convert_pascalvoc_dataset/build.py "/content/drive/MyDrive/Colab Notebooks/Sunshine Tech/Annotations" "/content/drive/MyDrive/Colab Notebooks/Sunshine Tech/data" 
```

2/ Training

+ Thay đổi đường dẫn dữ liệu đầu vào:

Tại file SSD/ssd/config/path_catalog.py:

Thay đổi my_root trong hàm get của "my_custom" thành đường dẫn gốc đến dữ liệu, thay đổi đường dẫn của DATASET

+ Thay đổi file config tại SSD/config:

NUM_CLASSES: số lượng object training (5 class bao gồm background)

DATASETS: kiểm tra tại file path_catalog.py để ghi 2 file tương ứng

OUTPUT_DIR: đường dẫn lưu model của chương trình (lưu ý, nếu đã train trước đó, thì cần phải thay đổi path trong file models/mobilenetv2_ssd…_my_dataset/last_checkpoint.txt với đường dẫn mới

+ Training stage:
```text
!python train.py --config-file configs/my_custom_config_320.yaml
```
3/ Testing
```text
!python test.py --config-file configs/my_custom_config_320.yaml
```
4/ Demo

!python --config-file <config file> --images_dir <image direction> --ckpt <model> --output_dir <output direction> --score_threshold <score>
```text
!python demo.py --config-file configs/my_custom_config_vgg_512.yaml --images_dir "/content/drive/MyDrive/Colab Notebooks/Sunshine Tech/cmnd_back" --ckpt "/content/drive/MyDrive/Colab Notebooks/Sunshine Tech/models/vgg_ssd512_30k_4cls/model_final.pth" --output_dir "/content/drive/MyDrive/Colab Notebooks/Sunshine Tech/demo" --score_threshold 0.5
```

## Troubleshooting
If you have issues running or compiling this code, we have compiled a list of common issues in [TROUBLESHOOTING.md](TROUBLESHOOTING.md). If your issue is not present there, please feel free to open a new issue.

## Citations
If you use this project in your research, please cite this project.
```text
@misc{lufficc2018ssd,
    author = {Congcong Li},
    title = {{High quality, fast, modular reference implementation of SSD in PyTorch}},
    year = {2018},
    howpublished = {\url{https://github.com/lufficc/SSD}}
}
```
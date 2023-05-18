# [스티칭] AB3DMOT 사용방법

## 환경세팅

**Requirements**

```bash
docker pull yckimm/stitching:v1.1 # (도커 공유)
```

## 디렉토리 세팅

data
|— stitch # 기존 데이터 폴더
|    |— data
|    |—    |— maps (사용하지 않음)
|    |—    |— produced    |— results    |— detection    |— centerpoint_val_H1 / results_val.json
|    |—    |— samples (ln -s stitch_centerpoint/stitch/samples ./)
|    |—    |— v1.0-trainval (ln -s stitch/v0.7-stitch ./v1.0-trainval)
|    |— nuKITTI    |— object    |— produced    |— correspondence


mv work_dirs/{model_name}/infos_val_01sweeps_withvelo_filter_True.json → AB3DMOT/data/stitch/data/produced/results/detection/centerpoint_val_H1/results_val.json

## 준비과정

cd path/to/AB3DMOT
pip3 install -r requirements.txt
git clone https://github.com/xinshuoweng/Xinshuo_PyToolbox
cd Xinshuo_PyToolbox
pip3 install -r requirements.txt
cd ..
git clone https://github.com/nutonomy/nuscenes-devkit.git

vim ~/.bashrc
export PYTHONPATH=${PYTHONPATH}:'/path/to/AB3DMOT'
export PYTHONPATH=${PYTHONPATH}:'/path/to/AB3DMOT/Xinshuo_PyToolbox'
export PYTHONPATH=${PYTHONPATH}:'/path/to/AB3DMOT/nuscenes-devkit/python-sdk'
source ~/.bashrc

sudo apt-get update
sudo apt-get install python3-tk
pip install opencv-python
apt update && apt install -y libsm6 libxext6
apt-get install -y libxrender-dev
pip install filterpy


cd AB3DMOT
mkdir -p data/stitch/data/produced/results/detection/centerpoint_val_H1
mkdir -p data/stitch/nuKITTI/object/produced/correspondence/


**실행해야할 명령어**

```bash


python3 scripts/nuScenes/export_kitti.py nuscenes_gt2kitti_obj --nusc_kitti_root ./data/stitch/nuKITTI --data_root ./data/stitch/data --result_root ./results/stitch --result_name centerpoint_val_H1 --split val

python3 scripts/nuScenes/export_kitti.py nuscenes_gt2kitti_trk --nusc_kitti_root ./data/stitch/nuKITTI --data_root ./data/stitch/data --result_root ./results/stitch --result_name centerpoint_val_H1 --split val

mkdir -p data/stitch/nuKITTI/object/produced/correspondence/

cp -r data/stitch/nuKITTI/object/val/correspondence.txt data/stitch/nuKITTI/object/produced/correspondence/val.txt

mkdir -p ./data/stitch/data/produced/results/detection/centerpoint_val_H1
cp -r ../../s14/AB3DMOT/data/stitch/data/produced/results/detection/centerpoint_val_H1/results_val_track.json ./data/stitch/data/produced/results/detection/centerpoint_val_H1/

python3 scripts/nuScenes/export_kitti.py nuscenes_obj_result2kitti --nusc_kitti_root ./data/stitch/nuKITTI --data_root ./data/stitch/data --result_root ./results/stitch --result_name centerpoint_val_H1 --split val

python3 scripts/pre_processing/convert_det2input.py --dataset stitch --split val --det_name centerpoint_val_H1

python3 main.py --dataset stitch --det_name centerpoint_val_H1 --split val


python3 scripts/nuScenes/export_kitti.py kitti_trk_result2nuscenes --nusc_kitti_root ./data/stitch/nuKITTI --data_root ./data/stitch/data --result_root ./results/stitch --result_name centerpoint_val_H1_val_H1 --split val

python3 scripts/post_processing/trk_conf_threshold.py --dataset stitch --result_sha centerpoint_val_H1_val_H1

python3 scripts/post_processing/visualization.py --dataset stitch --result_sha centerpoint_val_H1_val_H1_thres --split val --dataset stitch --split val
```


주의사항

results.json과 v1.0-trainval이 같은 버전을 가리키는지 확인해야함

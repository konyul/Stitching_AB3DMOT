# [스티칭] AB3DMOT 사용방법

## 환경세팅

**Requirements**

```bash
docker pull kyparkk/stitching_ab3dmot:v1.1 # (도커 공유)
docker run -it --gpus all --shm-size=512g kyparkk/stitching_ab3dmot:v1.1 /bin/bash
git clone https://github.com/konyul/stitching_AB3DMOT.git
```

## Quick demo
새로운 Detection 결과를 통해 Tracking 데모를 확인할 경우 아래의 순서로 진행 (10번 서버에서 진행)
```bash
docker exec -it stitching /bin/bash
mv work_dirs/{model_name}/infos_val_01sweeps_withvelo_filter_True.json → AB3DMOT/data/stitch/data/produced/results/detection/centerpoint_val_H1/results_val.json
```
아래의 **실행해야할 명령어** 를 순차적으로 입력

## 디렉토리 세팅

```bash
data
|— stitch # 기존 데이터 폴더
|    |— data
|    |—    |— maps (사용하지 않음)
|    |—    |— produced    |— results    |— detection    |— centerpoint_val_H1 (폴더 생성)
|    |—    |— samples (ln -s stitch_centerpoint/stitch/samples ./)
|    |—    |— v1.0-trainval (ln -s stitch/v0.x-stitch ./v1.0-trainval)
|    |— nuKITTI    |— object    |— produced    |— correspondence (폴더 생성)

해당 데이터셋은 10번서버의 경우 /mnt/sda/konyul/project/stitching/data/stitch에 존재
```

## detection result 경로 변경
```bash
mv work_dirs/{model_name}/infos_val_01sweeps_withvelo_filter_True.json → AB3DMOT/data/stitch/data/produced/results/detection/centerpoint_val_H1/results_val.json

현재 detection_results는 10번서버 /mnt/sda/konyul/project/stitching/data/stitch/data/produced/results/detection/centerpoint_val_H1/results_val.json 에 존재

```
## 준비과정

```bash
cd path/to/stitching_AB3DMOT
pip3 install -r requirements.txt
git clone https://github.com/xinshuoweng/Xinshuo_PyToolbox
cd Xinshuo_PyToolbox
pip3 install -r requirements.txt
cd ..
git clone https://github.com/nutonomy/nuscenes-devkit.git

vim ~/.bashrc
export PYTHONPATH=${PYTHONPATH}:'/path/to/stitching_AB3DMOT'
export PYTHONPATH=${PYTHONPATH}:'/path/to/stitching_AB3DMOT/Xinshuo_PyToolbox'
export PYTHONPATH=${PYTHONPATH}:'/path/to/stitching_AB3DMOT/nuscenes-devkit/python-sdk'
source ~/.bashrc

cd path/to/stitching_AB3DMOT
mkdir -p data/stitch/data/produced/results/detection/centerpoint_val_H1
mkdir -p data/stitch/nuKITTI/object/produced/correspondence/
```

**실행해야할 명령어**

```bash

*nuScenes GT to kitti detection format*
python3 scripts/nuScenes/export_kitti.py nuscenes_gt2kitti_obj --nusc_kitti_root ./data/stitch/nuKITTI --data_root ./data/stitch/data --result_root ./results/stitch --result_name centerpoint_val_H1 --split val

*nuScenes GT to kitti tracking format*
python3 scripts/nuScenes/export_kitti.py nuscenes_gt2kitti_trk --nusc_kitti_root ./data/stitch/nuKITTI --data_root ./data/stitch/data --result_root ./results/stitch --result_name centerpoint_val_H1 --split val

mkdir -p data/stitch/nuKITTI/object/produced/correspondence/
cp -r data/stitch/nuKITTI/object/val/correspondence.txt data/stitch/nuKITTI/object/produced/correspondence/val.txt

*nuscenes detection results to kitti format*
python3 scripts/nuScenes/export_kitti.py nuscenes_obj_result2kitti --nusc_kitti_root ./data/stitch/nuKITTI --data_root ./data/stitch/data --result_root ./results/stitch --result_name centerpoint_val_H1 --split val

*detection results to tracking input*
python3 scripts/pre_processing/convert_det2input.py --dataset stitch --split val --det_name centerpoint_val_H1

*run AB3DMOT*
python3 main.py --dataset stitch --det_name centerpoint_val_H1 --split val

*tracking results to nuscenes format* (path_to_stitching_AB3DMOT/results/stitch/{result_name}/results_val.json)
python3 scripts/nuScenes/export_kitti.py kitti_trk_result2nuscenes --nusc_kitti_root ./data/stitch/nuKITTI --data_root ./data/stitch/data --result_root ./results/stitch --result_name centerpoint_val_H1_val_H1 --split val

*post_processing*
python3 scripts/post_processing/trk_conf_threshold.py --dataset stitch --result_sha centerpoint_val_H1_val_H1

*visualization* (path_to_stitching_AB3DMOT/results/stitch/{result_sha})
python3 scripts/post_processing/visualization.py --dataset stitch --result_sha centerpoint_val_H1_val_H1_thres --split val --dataset stitch --split val
```


주의사항

results.json과 v1.0-trainval이 같은 버전을 가리키는지 확인해야함

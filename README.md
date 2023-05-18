Stitch 명령어 정리

python3 scripts/nuScenes/export_kitti.py nuscenes_gt2kitti_obj --nusc_kitti_root ./data/stitch/nuKITTI --data_root ./data/stitch/data --result_root ./results/stitch --result_name centerpoint_val_H1 --split val

python3 scripts/nuScenes/export_kitti.py nuscenes_gt2kitti_trk --nusc_kitti_root ./data/stitch/nuKITTI --data_root ./data/stitch/data --result_root ./results/stitch --result_name centerpoint_val_H1 --split val

mkdir -p data/stitch/nuKITTI/object/produced/correspondence/

cp -r data/stitch/nuKITTI/object/val/correspondence.txt data/stitch/nuKITTI/object/produced/correspondence/val.txt

mkdir -p ./data/stitch/data/produced/results/detection/centerpoint_val_H1
cp -r ../../s14/AB3DMOT/data/stitch/data/produced/results/detection/centerpoint_val_H1/results_val_track.json ./data/stitch/data/produced/results/detection/centerpoint_val_H1/

python3 scripts/nuScenes/export_kitti.py nuscenes_obj_result2kitti --nusc_kitti_root ./data/stitch/nuKITTI --data_root ./data/stitch/data --result_root ./results/stitch --result_name centerpoint_val_H1 --split val
여기에서 box_to_string할때 quaternion to rad할때 문제발생했음

python3 scripts/pre_processing/convert_det2input.py --dataset stitch --split val --det_name centerpoint_val_H1

python3 main.py --dataset stitch --det_name centerpoint_val_H1 --split val
여기서 성능나옴


python3 scripts/nuScenes/export_kitti.py kitti_trk_result2nuscenes --nusc_kitti_root ./data/stitch/nuKITTI --data_root ./data/stitch/data --result_root ./results/stitch --result_name centerpoint_val_H1_val_H1 --split val

python3 scripts/post_processing/trk_conf_threshold.py --dataset stitch --result_sha centerpoint_val_H1_val_H1

python3 scripts/post_processing/visualization.py --dataset stitch --result_sha centerpoint_val_H1_val_H1_thres --split val --dataset stitch --split val

주의사항

results.json과 v1.0-trainval이 같은 버전을 가리키는지 확인해야함

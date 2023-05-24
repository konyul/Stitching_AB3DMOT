# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.

from typing import Dict, List

from nuscenes import NuScenes


train_track = \
    [ 'scene_060', 'scene_057', 'scene_045', 'scene_041','scene_037',
        'scene_034', 'scene_068', 'scene_029', 'scene_082', 'scene_009',
        'scene_030', 'scene_010', 'scene_055', 'scene_054', 'scene_059',
        'scene_044', 'scene_050', 'scene_061', 'scene_066', 'scene_072',
        'scene_043', 'scene_006', 'scene_002', 'scene_046', 'scene_016',
        'scene_026', 'scene_031', 'scene_063', 'scene_021', 'scene_074',
        'scene_042', 'scene_011', 'scene_033', 'scene_005', 'scene_032',
        'scene_013', 'scene_081', 'scene_056', 'scene_001', 'scene_069',
        'scene_004', 'scene_049', 'scene_073', 'scene_023',
        'scene_053', 'scene_079', 'scene_080', 'scene_058']
train_detect = \
    ['scene_007', 'scene_020', 'scene_064', 'scene_017',
            'scene_065', 'scene_077', 'scene_051', 'scene_024',
            'scene_018', 'scene_047', 'scene_027', 'scene_071',
            'scene_022', 'scene_015', 'scene_014', 'scene_019', 'scene_078', 'scene_052']
train = list(sorted(set(train_detect + train_track)))

val = \
    ['scene_012', 'scene_067', 'scene_038', 'scene_035', 'scene_076',
            'scene_008', 'scene_048', 'scene_075', 'scene_025', 'scene_028',
            'scene_036', 'scene_039', 'scene_062', 'scene_070', 'scene_003', 'scene_040']
val_track=\
        ['scene_003', 'scene_008', 'scene_012']
val_track2= ['scene_001', 'scene 002', 'scene_004', 'scene_005', 'scene_006',
             'scene 007', 'scene_009', 'scene_010', 'scene_011', 'scene_013',
             'scene_014','scene_ 015', 'scene_016', 'scene_017', 'scene_018'
             'scene_019', 'scene_020']
val = list(sorted(set(val + val_track)))
stitch = val
test = \
    ['scene-0077', 'scene-0078', 'scene-0079', 'scene-0080', 'scene-0081', 'scene-0082', 'scene-0083', 'scene-0084',
     'scene-0085', 'scene-0086', 'scene-0087', 'scene-0088', 'scene-0089', 'scene-0090', 'scene-0091', 'scene-0111',
     'scene-0112', 'scene-0113', 'scene-0114', 'scene-0115', 'scene-0116', 'scene-0117', 'scene-0118', 'scene-0119',
     'scene-0140', 'scene-0142', 'scene-0143', 'scene-0144', 'scene-0145', 'scene-0146', 'scene-0147', 'scene-0148',
     'scene-0265', 'scene-0266', 'scene-0279', 'scene-0280', 'scene-0281', 'scene-0282', 'scene-0307', 'scene-0308',
     'scene-0309', 'scene-0310', 'scene-0311', 'scene-0312', 'scene-0313', 'scene-0314', 'scene-0333', 'scene-0334',
     'scene-0335', 'scene-0336', 'scene-0337', 'scene-0338', 'scene-0339', 'scene-0340', 'scene-0341', 'scene-0342',
     'scene-0343', 'scene-0481', 'scene-0482', 'scene-0483', 'scene-0484', 'scene-0485', 'scene-0486', 'scene-0487',
     'scene-0488', 'scene-0489', 'scene-0490', 'scene-0491', 'scene-0492', 'scene-0493', 'scene-0494', 'scene-0495',
     'scene-0496', 'scene-0497', 'scene-0498', 'scene-0547', 'scene-0548', 'scene-0549', 'scene-0550', 'scene-0551',
     'scene-0601', 'scene-0602', 'scene-0603', 'scene-0604', 'scene-0606', 'scene-0607', 'scene-0608', 'scene-0609',
     'scene-0610', 'scene-0611', 'scene-0612', 'scene-0613', 'scene-0614', 'scene-0615', 'scene-0616', 'scene-0617',
     'scene-0618', 'scene-0619', 'scene-0620', 'scene-0621', 'scene-0622', 'scene-0623', 'scene-0624', 'scene-0827',
     'scene-0828', 'scene-0829', 'scene-0830', 'scene-0831', 'scene-0833', 'scene-0834', 'scene-0835', 'scene-0836',
     'scene-0837', 'scene-0838', 'scene-0839', 'scene-0840', 'scene-0841', 'scene-0842', 'scene-0844', 'scene-0845',
     'scene-0846', 'scene-0932', 'scene-0933', 'scene-0935', 'scene-0936', 'scene-0937', 'scene-0938', 'scene-0939',
     'scene-0940', 'scene-0941', 'scene-0942', 'scene-0943', 'scene-1026', 'scene-1027', 'scene-1028', 'scene-1029',
     'scene-1030', 'scene-1031', 'scene-1032', 'scene-1033', 'scene-1034', 'scene-1035', 'scene-1036', 'scene-1037',
     'scene-1038', 'scene-1039', 'scene-1040', 'scene-1041', 'scene-1042', 'scene-1043']

mini_train = \
    ['scene_1002', 'scene_1005', 'scene_1008',]

mini_val = \
    ['scene-1003']


def create_splits_logs(split: str, nusc: 'NuScenes') -> List[str]:
    """
    Returns the logs in each dataset split of nuScenes.
    Note: Previously this script included the teaser dataset splits. Since new scenes from those logs were added and
          others removed in the full dataset, that code is incompatible and was removed.
    :param split: NuScenes split.
    :param nusc: NuScenes instance.
    :return: A list of logs in that split.
    """
    # Load splits on a scene-level.

    scene_splits = create_splits_scenes(verbose=False)

    assert split in scene_splits.keys(), 'Requested split {} which is not a known nuScenes split.'.format(split)

    # Check compatibility of split with nusc_version.
    version = nusc.version
    if split in {'stitch'}:
        assert version.endswith('stitch'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(split, version)
    elif split in {'train', 'val', 'train_detect', 'train_track', 'val_track'}:
        assert version.endswith('trainval'), \
            'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)
    elif split in {'mini_train', 'mini_val'}:
        assert version.endswith('mini'), \
            'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)
    elif split == 'test':
        assert version.endswith('test'), \
            'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)
    else:
        raise ValueError('Requested split {} which this function cannot map to logs.'.format(split))

    # Get logs for this split.
    #scene_to_log = {scene['name']: nusc.get('log', scene['log_token'])['logfile'] for scene in nusc.scene}
    scene_to_log = {scene['name']: nusc.get("sample_data",nusc.get("sample",scene['first_sample_token'])['data']['LIDAR_TOP'])['filename'].split("/")[-1].split(".")[0][:-3] for scene in nusc.scene}
    logs = set()
    scenes = scene_splits[split]
    for scene in scenes:
        logs.add(scene_to_log[scene])

    return list(logs)


def create_splits_scenes(verbose: bool = False) -> Dict[str, List[str]]:
    """
    Similar to create_splits_logs, but returns a mapping from split to scene names, rather than log names.
    The splits are as follows:
    - train/val/test: The standard splits of the nuScenes dataset (700/150/150 scenes).
    - mini_train/mini_val: Train and val splits of the mini subset used for visualization and debugging (8/2 scenes).
    - train_detect/train_track: Two halves of the train split used for separating the training sets of detector and
        tracker if required.
    :param verbose: Whether to print out statistics on a scene level.
    :return: A mapping from split name to a list of scenes names in that split.
    """
    # Use hard-coded splits.
    all_scenes = train + val
    # import pdb; pdb.set_trace()
    assert len(all_scenes) == 82 and len(set(all_scenes)) == 82, 'Error: Splits incomplete!'
    scene_splits = {'train': train, 'val': val, 'train_detect': train_detect, 'train_track': train_track, 'stitch': stitch, 'val_track': val_track}

    # Optional: Print scene-level stats.
    if verbose:
        for split, scenes in scene_splits.items():
            print('%s: %d' % (split, len(scenes)))
            print('%s' % scenes)

    return scene_splits


if __name__ == '__main__':
    # Print the scene-level stats.
    create_splits_scenes(verbose=True)

conf = {
    'cambridge': [
        {
            # Set N.1
            'beams': 2,
            'steps': 40,
            'N': 52,
            'M': 2,
            'feat_model': 'DepthV2',
            'protocol': '2_1',
            'center_std': [1.5,1.5, 1.5],
            'teta': [2],
            'gamma': 0.3,
            'res': 320,
            'colmap_res': 320,
            'feat_level' : [8],
            'foundation_model_path': 'ckpt/dinov2_vitb14_pretrain.pth',
        },      
    ]
}


def get_config(ds_name):
    cambridge_scenes = [
        'StMarysChurch', 'OldHospital', 'KingsCollege', 'ShopFacade','inTraj', 'outTraj','Synthesis', 'Swiss_in', 'Swiss_out', 'Video'
    ]
    
    if ds_name in cambridge_scenes:
        return conf['cambridge']
    else:
        return NotImplementedError

b = 'C:\\Users\\dbouchie\\Downloads\\TinyActions-main\\TinyActions-main\\'
#b = ''

def build_config(dataset):
    cfg = type('', (), {})()
    cfg.data_folder = b + 'TinyVIRAT-v2/videos'
    cfg.train_annotations = b + 'TinyVIRAT-v2/tiny_train_v2.json'
    cfg.val_annotations = b + 'TinyVIRAT-v2/tiny_val_v2.json'
    cfg.test_annotations = b + 'TinyVIRAT-v2/tiny_test_v2_public.json'
    cfg.class_map = b + 'TinyVIRAT-v2/class_map.json'
    cfg.pose_folder = b + 'TinyVIRAT-v2/pose_high_confidence'
    cfg.num_classes = 26
    #cfg.saved_models_dir = './results/saved_models'
    #cfg.tf_logs_dir = './results/logs'
    return cfg

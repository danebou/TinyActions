def build_config(dataset):
    path = 'C:\\Users\\dbouchie\\Downloads\\TinyActions-main\\TinyActions-main\\TinyVIRAT-v2'
    cfg = type('', (), {})()
    if dataset == 'TinyVirat' or True:
        cfg.data_folder = path + '/videos'
        cfg.train_annotations = path + '/tiny_train_v2.json'
        cfg.val_annotations = path + '/tiny_val_v2.json'
        cfg.test_annotations = path + '/tiny_test_v2_public.json'
        cfg.class_map = path + '/class_map.json'
        cfg.flow_folder = path + '/optical_flow'
        cfg.num_classes = 26
    # elif dataset == 'TinyVirat-d':
    #     cfg.data_folder = 'datasets/TinyVIRAT-v2/videos'
    #     cfg.train_annotations = 'datasets/TinyVIRAT-v2/tiny_train_v2.json'
    #     cfg.val_annotations = 'datasets/TinyVIRAT-v2/tiny_val_v2.json'
    #     cfg.test_annotations = 'datasets/TinyVIRAT-v2/tiny_test_v2_public.json'
    #     cfg.class_map = 'datasets/TinyVIRAT-v2/class_map.json'
    #     cfg.stabilize_folder = 'datasets/TinyVIRAT-v2/virat_stabilize'
    #     cfg.num_classes = 26
    #cfg.saved_models_dir = './results/saved_models'
    #cfg.tf_logs_dir = './results/logs'
    return cfg

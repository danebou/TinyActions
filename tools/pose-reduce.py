import os
import json

virat_folder = 'TinyVIRAT-v2\\pose_high_confidence'
virat_files = [(f, os.path.join(dp, f)) for dp, dn, fn in os.walk(os.path.expanduser(virat_folder)) for f in fn]

file_count = len(virat_files)

for i, (file_name, file_path) in enumerate(virat_files):
    if not file_name.endswith(".json"): continue
    file_rel_path = file_path.replace(virat_folder, "")
    target_file = f'TinyVIRAT-v2\\pose_high_confidence_top10\\{file_rel_path}'
    target_dir = target_file.replace(file_name, "")
    #target_file = target_file.replace('.mp4', '.json')
    print(f'File {i}/{file_count} {target_file}')
    if os.path.isfile(target_file): continue

    os.makedirs(target_dir, exist_ok=True)


    with open(file_path, 'r') as f:
        pose_json = json.load(f)

    pose_data = {}
    for p in pose_json:
        id = int(p['image_id'][:-len('.jpg')])
        if not id in pose_data:
            pose_data[id] = []
        pose_data[id].append(p)

    # Get top pose_data based on score
    for k,v in pose_data.items():
        pose_data[k] = sorted(v, key = lambda x: x['score'], reverse=True)[:10]

    final_json = []
    for frame_id,pose in pose_data.items():
        for p in pose:
            final_json.append(p)

    with open(target_file, 'w') as f:
        json.dump(final_json, f)

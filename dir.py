import os
IMAGE_KEYFRAME_PATH = r"/aic/challenge_data"
VISUAL_FEATURES_PATH = r"/aic/challenge_data"

for folder_path in os.listdir(IMAGE_KEYFRAME_PATH):
  folder_map = os.path.join(IMAGE_KEYFRAME_PATH,folder_path)
  for mapp in os.listdir(folder_map):
    if mapp == 'keyframes':
      keyframes_map = os.path.join(folder_map, mapp)
      for keyframes_dir in os.listdir(keyframes_map):
        img_paths = os.path.join(keyframes_map, keyframes_dir)
    elif mapp == 'clip-features':
      feature_paths = os.path.join(folder_map, mapp)

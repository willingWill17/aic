import os
IMAGE_KEYFRAME_PATH = r"home/mmlab/challenge_data"
VISUAL_FEATURES_PATH = r"home/mmlab/challenge_data"

for folder_path3 in os.listdir(IMAGE_KEYFRAME_PATH):
  folder_path3 = os.path.join(IMAGE_KEYFRAME_PATH, folder_path3)
  feature_path2 = os.path.join(VISUAL_FEATURES_PATH, folder_path3)
  feature_path = os.path.join(feature_path2, 'clip-features')
  folder_path2 = os.path.join(folder_path3, 'keyframes')
  for img_paths in os.listdir(folder_path2):
    img_paths = os.path.join(folder_path2, img_paths)

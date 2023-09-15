import os
IMAGE_KEYFRAME_PATH = "/content/drive/MyDrive/HCMAI22_MiniBatch1/Keyframes/"
VISUAL_FEATURES_PATH = "/content/drive/MyDrive/HCMAI22_MiniBatch1/CLIP_features"

for folder_path2 in os.listdir(IMAGE_KEYFRAME_PATH):
  folder_path2 = os.path.join(IMAGE_KEYFRAME_PATH, folder_path2)  
  for folder_path in os.listdir(folder_path2):
    folder_path = os.path.join(folder_path2, folder_path)
    for img_paths in os.listdir(folder_path):
      print(img_paths)
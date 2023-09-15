import os
import numpy as np

from tqdm import tqdm
from PIL import Image

import requests

import torch
import clip
from dir import folder_path, img_paths
IMAGE_KEYFRAME_PATH = "/content/drive/MyDrive/HCMAI22_MiniBatch1/Keyframes/"
VISUAL_FEATURES_PATH = "/content/drive/MyDrive/HCMAI22_MiniBatch1/CLIP_features"
class TextEmbedding():
  def __init__(self):
    self.device = device
    self.model = model
    self.preprocess = preprocess

  def __call__(self, text: str) -> np.ndarray:
    text_inputs = clip.tokenize([text]).to(self.device)
    with torch.no_grad():
        text_feature = self.model.encode_text(text_inputs)[0]

    return text_feature.detach().cpu().numpy()


# ==================================
text = "2 people are standing"
text_embedd = TextEmbedding()
text_feat_arr = text_embedd(text)
np.save(VISUAL_FEATURES_PATH, text_feat_arr, allow_pickle=True, fix_imports=True)
print(text_feat_arr.shape, type(text_feat_arr))


for images in os.listdir(folder_path):
      # Step 1: Load and Preprocess Images
      # For simplicity, let's assume you have a list of file paths to your images
    images = []
    images_error = []
    for path in img_paths:
        try:
          img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale for simplicity
          img = cv2.resize(img, (64, 64))  # Resize to a consistent size
          images.append(img)
          # PIL.Image.open(path)
        except Exception as e:
          images_error.append(path)
          print(e)
        # Step 2: Vectorize the Images
        vectorized_images = [img.flatten() for img in images]

        # Step 3: Store in a Numpy File
        np.save('vectorized_images.npy', np.array(vectorized_images))

        # Optionally, you can later load the numpy file using np.load('vectorized_images.npy')
        images = os.path.join(folder_path, images)
        img_paths.append(images)
        from typing import List, Tuple
def indexing_methods() -> List[Tuple[str, int, np.ndarray],]:
    db = []
    '''Duyệt tuần tự và đọc các features vector từ file .npy'''
    for feat_npy in tqdm(os.listdir(VISUAL_FEATURES_PATH)):
        video_name = feat_npy.split('.')[0]
        feats_arr = np.load(os.path.join(VISUAL_FEATURES_PATH, feat_npy), allow_pickle=True)
    for idx, feat in enumerate(feats_arr):
      '''Lưu mỗi records với 3 trường thông tin là video_name, keyframe_id, feature_of_keyframes'''
        instance = (video_name, idx, feat)
        db.append(instance)
    return db


# ==================================
visual_features_db = indexing_methods()
print()
print(visual_features_db[0][:2], visual_features_db[0][-1].shape)
def search_engine(query_arr: np.array,
                  db: list,
                  topk:int=10,
                  measure_method: str="dot_product") -> List[dict,]:

  '''Duyệt tuyến tính và tính độ tương đồng giữa 2 vector'''
  measure = []
  for ins_id, instance in enumerate(db):
    video_name, idx, feat_arr = instance

    if measure_method=="dot_product":
      distance = query_arr @ feat_arr.T
    measure.append((ins_id, distance))

  '''Sắp xếp kết quả'''
  measure = sorted(measure, key=lambda x:x[-1], reverse=True)

  '''Trả về top K kết quả'''
  search_result = []
  for instance in measure[:topk]:
    ins_id, distance = instance
    video_name, idx, _ = db[ins_id]

    search_result.append({"video_name":video_name,
                          "keyframe_id": idx,
                          "score": distance})
  return search_result


# ==================================
search_result = search_engine(text_feat_arr, visual_features_db, 10)
print(search_result)
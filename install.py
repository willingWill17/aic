import os
import numpy as np

from tqdm import tqdm
from PIL import Image

import requests
import cv2
import torch
import clip
from dir import folder_path2, img_paths, feature_path
IMAGE_KEYFRAME_PATH = r"/aic/challenge_data"
VISUAL_FEATURES_PATH = r"/aic/challenge_data"

class TextEmbedding():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Assuming you have a model initialized, you would do something like:
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def __call__(self, query: str) -> np.ndarray:
        # Assuming clip.tokenize is a valid function for tokenization
        query_inputs = clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            query_feature = self.model.encode_text(query_inputs)[0]

        return query_feature.detach().cpu().numpy()


# ==================================
querry = input("Enter your querry: ")
querry_embedd = TextEmbedding()
querry_feat_arr = querry_embedd(querry)
np.save(VISUAL_FEATURES_PATH, querry_feat_arr, allow_pickle=True, fix_imports=True)
print(querry_feat_arr.shape, type(querry_feat_arr))


# Optionally, you can later load the numpy file using np.load('vectorized_images.npy')
from typing import List, Tuple
def indexing_methods() -> List[Tuple[str, int, np.ndarray],]:
    db = []
    '''Duyệt tuần tự và đọc các features vector từ file .npy'''
    for feat_npy in tqdm(os.listdir(feature_path)):
      video_name = feat_npy.split('.')[0]
      feats_arr = np.load(os.path.join(feature_path , feat_npy), allow_pickle=True)
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
search_result = search_engine(querry_feat_arr, visual_features_db, 10)
print(search_result)

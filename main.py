# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import cv2
import numpy as np
import os
import tensorflow as tf
import pandas as pd
import sys

## import the handfeature extractor class

from handshape_feature_extractor import HandShapeFeatureExtractor

def cosine_sim(vec1: np.ndarray, vec2: np.ndarray):
  dot = np.dot(vec1, vec2.T)
  mag_1 = np.linalg.norm(vec1)
  mag_2= np.linalg.norm(vec2)
  
  if mag_1 == 0 or mag_2 == 0:
    return 0
  
  return dot / (mag_1 * mag_2)


def get_gesture_label(filename: str):
    # Define a lookup table of known gestures
    gestures = {
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "DecreaseFanSpeed", "FanOff", "FanOn", "IncreaseFanSpeed",
        "LightOff", "LightOn", "SetThermo"
    }
    
    # Remove extension
    base_name = filename.rsplit('.', 1)[0]  # removes .mp4 or any other extension
    # Split by '-' and take the last part
    label_candidate = base_name.split('-')[-1]
    
    # Check if candidate is a known gesture
    if label_candidate in gestures:
        return label_candidate
    else:
        # raise BaseException(f"Not found {filename}")
        # print(f"NOt found: {filename}")
        # return None  # not found
        return "0"
    
label_lookup = {
  "0": "0",
  "1": "1",
  "2": "2",
  "3": "3",
  "4": "4",
  "5": "5",
  "6": "6",
  "7": "7",
  "8": "8",
  "9": "9",
  "DecreaseFanSpeed": "10",
  "FanOff": "11",
  "FanOn": "12",
  "IncreaseFanSpeed": "13",
  "LightOff": "14",
  "LightOn": "15",
  "SetThermo": "16",
}

# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video

BASE = os.path.dirname(os.path.abspath(__file__))

files = os.listdir(os.path.join(BASE, 'traindata'))
print(files)

extractor = HandShapeFeatureExtractor()

training_data: dict[str, np.ndarray] = {}

for file in files:
  path = os.path.join(BASE, 'traindata', file)
  print(f"Reading {path}")
  cap = cv2.VideoCapture(path)

  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  middle_frame_index = total_frames // 2

  cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
  ret, frame = cap.read()
  if not ret:
    print("No ret")
    continue
  
  cv2.imwrite("./test.jpg", frame)
  cap.release()

  vec = extractor.extract_feature(frame)
  # print(vec)
  
  training_data[path] = vec
  # input()

print(training_data)

# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here 
# Extract the middle frame of each gesture video

test_data: dict[str, np.ndarray] = {}
files = os.listdir(os.path.join(BASE, 'test'))
for file in files:
  path = os.path.join(BASE, 'test', file)
  print(f"Reading {path}")
  cap = cv2.VideoCapture(path)

  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  middle_frame_index = total_frames // 2

  cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
  ret, frame = cap.read()
  if not ret:
    print("No ret")
    continue
  
  cv2.imwrite("./test.jpg", frame)
  cap.release()

  vec = extractor.extract_feature(frame)
  # print(vec)
  
  test_data[path] = vec
  # input()

df: pd.DataFrame = pd.DataFrame()

for test in test_data:
  accuracy = 0.0
  test_vid = ''
  for train in training_data:
    # TODO: add a "already found" to optimize for performance

    similarity = cosine_sim(training_data[train], test_data[test])
    if similarity > accuracy:
      accuracy = similarity
      test_vid = train
    # print(f'Simularity for {train} and {test} = {similarity}')

  print(f'Best match for {test} is {test_vid} with {accuracy} accuracy')
  
  gesture_name = get_gesture_label(os.path.basename(test))
  if gesture_name is None:
    print(f'Skipping file name: {test}, since its not found')
    continue

  label = label_lookup[gesture_name]
  if label is not None:
    df = pd.concat([df, pd.DataFrame([label])])

print(df.shape)
df = df.head(51)

df.to_csv("Results.csv", index=False)


# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================


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


gesture_lookup = {
  "T1-H-0.mp4": "0",
  "T1-H-1.mp4": "1",
  "T1-H-2.mp4": "2",
  "T1-H-3.mp4": "3",
  "T1-H-4.mp4": "4",
  "T1-H-5.mp4": "5",
  "T1-H-6.mp4": "6",
  "T1-H-7.mp4": "7",
  "T1-H-8.mp4": "8",
  "T1-H-9.mp4": "9",
  "T1-H-DecreaseFanSpeed.mp4": "DecreaseFanSpeed",
  "T1-H-FanOff.mp4": "FanOff",
  "T1-H-FanOn.mp4": "FanOn",
  "T1-H-IncreaseFanSpeed.mp4": "IncreaseFanSpeed",
  "T1-H-LightOff.mp4": "LightOff",
  "T1-H-LightOn.mp4": "LightOn",
  "T1-H-SetThermo.mp4": "SetThermo",
  "T2-H-0.mp4": "0",
  "T2-H-1.mp4": "1",
  "T2-H-2.mp4": "2",
  "T2-H-3.mp4": "3",
  "T2-H-4.mp4": "4",
  "T2-H-5.mp4": "5",
  "T2-H-6.mp4": "6",
  "T2-H-7.mp4": "7",
  "T2-H-8.mp4": "8",
  "T2-H-9.mp4": "9",
  "T2-H-DecreaseFanSpeed.mp4": "DecreaseFanSpeed",
  "T2-H-FanOff.mp4": "FanOff",
  "T2-H-FanOn.mp4": "FanOn",
  "T2-H-IncreaseFanSpeed.mp4": "IncreaseFanSpeed",
  "T2-H-LightOff.mp4": "LightOff",
  "T2-H-LightOn.mp4": "LightOn",
  "T2-H-SetThermo.mp4": "SetThermo",
  "T3-H-0.mp4": "0",
  "T3-H-1.mp4": "1",
  "T3-H-2.mp4": "2",
  "T3-H-3.mp4": "3",
  "T3-H-4.mp4": "4",
  "T3-H-5.mp4": "5",
  "T3-H-6.mp4": "6",
  "T3-H-7.mp4": "7",
  "T3-H-8.mp4": "8",
  "T3-H-9.mp4": "9",
  "T3-H-DecereaseFanSpeed.mp4": "DecreaseFanSpeed",
  "T3-H-FanOff.mp4": "FanOff",
  "T3-H-FanOn.mp4": "FanOn",
  "T3-H-IncreaseFanSpeed.mp4": "IncreaseFanSpeed",
  "T3-H-LightOff.mp4": "LightOff",
  "T3-H-LightOn.mp4": "LightOn",
  "T3-H-SetThermo.mp4": "SetThermo"
}

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

df: pd.DataFrame = pd.DataFrame({
  "Video": [],
  "Gesture Name": [],
  "Label": [],
}) 

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
  
  gesture_name = gesture_lookup[os.path.basename(test)]
  label = label_lookup[gesture_name]
  
  df = pd.concat([df, pd.DataFrame([{"Video": os.path.basename(test), "Gesture Name": gesture_name, "Label": label}])])

with open('results.csv', 'w') as f:
  df.to_csv(f, index=False)


# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================


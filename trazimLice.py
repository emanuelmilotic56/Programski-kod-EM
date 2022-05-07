#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:13:15 2022

@author: Emanuel Milotić
"""
#from gpiozero import Servo
#from gpiozero.pins.pigpio import PiGPIOFactory
#factory = PiGPIOFactory()
#servo1 = Servo(17, pin_factory=factory)

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Čekam video frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
          x0 = face_landmarks.landmark[0].x
          y0 = face_landmarks.landmark[0].y
          x1 = face_landmarks.landmark[1].x
          y1 = face_landmarks.landmark[1].y
          x = ((x0 +x1)/2)+1
          y = ((y0 +y1)/2)+1
          '''
          print ('X0= ', face_landmarks.landmark[0].x)
          print ('Y0= ', face_landmarks.landmark[0].y)
          print ('Z0= ', face_landmarks.landmark[0].z)
          print('---')
          print ('X1= ', face_landmarks.landmark[1].x)
          print ('Y1= ', face_landmarks.landmark[1].y)
          print ('Z1= ', face_landmarks.landmark[1].z)
          '''
          print('---')
          print('x = ',x)
          print('y = ', y)
          
          mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    
    if cv2.waitKey(5) & 0xFF == 27:

        break
cap.release()
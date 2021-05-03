import warnings
warnings.filterwarnings("ignore")
import time
import os
from os import path
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from deepface.basemodels import VGGFace
from deepface.extendedmodels import Emotion
from deepface.commons import functions, realtime, distance as dst


def build_model(model_name):
	
	"""
	This function builds a deepface model
	Parameters:
		model_name (string): face recognition or facial attribute model
			VGG-Face, Facenet, OpenFace, DeepFace, DeepID for face recognition
			Age, Gender, Emotion, Race for facial attributes
	
	Returns:
		built deepface model
	"""
	
	models = {
		'VGG-Face': VGGFace.loadModel,
		'Emotion': Emotion.loadModel
	}

	model = models.get(model_name)
	
	if model:
		model = model()
		#print('Using {} model backend'.format(model_name))
		return model
	else:
		raise ValueError('Invalid model_name passed - {}'.format(model_name))


def stream(db_path = '', model_name ='VGG-Face', distance_metric = 'cosine'
			, enable_face_analysis = True
			, source = 0, time_threshold = 1, frame_threshold = 1):
	
	"""
	This function applies real time face recognition and facial attribute analysis
	
	Parameters:
		db_path (string): facial database path. You should store some .jpg files in this folder.
		
		model_name (string): VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib or Ensemble
		
		distance_metric (string): cosine, euclidean, euclidean_l2
		
		enable_facial_analysis (boolean): Set this to False to just run face recognition
		
		source: Set this to 0 for access web cam. Otherwise, pass exact video path.
		
		time_threshold (int): how many second analyzed image will be displayed
		
		frame_threshold (int): how many frames required to focus on face
		
	"""
	
	if time_threshold < 1:
		raise ValueError("time_threshold must be greater than the value 1 but you passed "+str(time_threshold))
	
	if frame_threshold < 1:
		raise ValueError("frame_threshold must be greater than the value 1 but you passed "+str(frame_threshold))
		
	functions.initialize_detector(detector_backend = 'opencv')
	
	realtime.analysis(db_path, model_name, distance_metric, enable_face_analysis
						, source = source, time_threshold = time_threshold, frame_threshold = frame_threshold)

	
#---------------------------
#main

functions.initializeFolder()

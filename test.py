# coding:utf-8

from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import time
from PIL import Image
from datetime import timedelta
import numpy as np

img_path = '1.jpg'
img = Image.open(img_path)
img = img.resize((28, 28), Image.ANTIALIAS)
img_arr = np.array(img.convert('L'))
print(img_arr.shape)
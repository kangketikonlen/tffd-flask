import os
import werkzeug
import cv2

from flask import request
from flask_restx import reqparse
from .auth_helper import Auth

BASE_PATH = os.path.abspath(os.path.dirname("."))
UPLOAD_PATH = os.path.join(BASE_PATH, "uploads/")
OUTPUT_PATH = os.path.join(BASE_PATH, "outputs/")


def upload():
    parse = reqparse.RequestParser()
    parse.add_argument(
        "file", type=werkzeug.datastructures.FileStorage, location="files"
    )
    args = parse.parse_args()
    image_file = args["file"]
    ext = image_file.filename.split(".")[1]
    user = Auth.extract_user(request)
    filename = user.username + "." + ext
    image_file.save(UPLOAD_PATH + filename)
    split_images(user.username, filename)


def split_images(object, file):
    upload_dir = os.path.join(UPLOAD_PATH, file)
    output_dir = os.path.join(OUTPUT_PATH, object + "/")
    capture = cv2.VideoCapture(upload_dir)
    frameNr = 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while True:
        success, frame = capture.read()

        if success:
            original = f"{output_dir}{object}_{frameNr}.jpg"
            cv2.imwrite(original, frame)
            resize = cv2.imread(original, cv2.IMREAD_UNCHANGED)
            scale_percent = 40  # percent of original size
            width = int(resize.shape[1] * scale_percent / 100)
            height = int(resize.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(resize, dim, interpolation=cv2.INTER_AREA)
            cv2.imwrite(original, resized)
            print("File created :", original)
        else:
            break

        frameNr = frameNr + 1
    capture.release()

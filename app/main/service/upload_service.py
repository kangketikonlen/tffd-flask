import os
import werkzeug

from flask import request
from flask_restx import reqparse
from .auth_helper import Auth

BASE_PATH = os.path.abspath(os.path.dirname("."))


def upload():
    parse = reqparse.RequestParser()
    parse.add_argument(
        "file", type=werkzeug.datastructures.FileStorage, location="files"
    )
    args = parse.parse_args()
    image_file = args["file"]
    ext = image_file.filename.split(".")[1]
    UPLOAD_DIR = BASE_PATH + "/uploads/"
    user = Auth.extract_user(request)
    filename = user.username + "." + ext
    image_file.save(UPLOAD_DIR + filename)

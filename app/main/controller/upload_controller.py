from flask_restx import Resource

from app.main.util.decorator import token_required
from ..util.dto import UploadDto
from ..service.upload_service import upload

api = UploadDto.api


@api.route("/")
class Upload(Resource):
    @token_required
    def post(self):
        upload()

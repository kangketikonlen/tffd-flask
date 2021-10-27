from flask_restx import Resource

from app.main.util.decorator import token_required
from ..util.dto import TrainDto
from ..service.train_service import start

api = TrainDto.api


@api.route("/")
class Train(Resource):
    @token_required
    def post(self):
        result = start()
        response = {
            "status": "success",
            "message": "Predicted "+result
        }
        return response, 200

from flask_restx import Api
from flask import Blueprint

from .main.controller.user_controller import api as user_ns
from .main.controller.auth_controller import api as auth_ns
from .main.controller.upload_controller import api as upload_ns
from .main.controller.train_controller import api as train_ns

blueprint = Blueprint("api", __name__)
authorizations = {"apikey": {"type": "apiKey", "in": "header", "name": "Authorization"}}

api = Api(
    blueprint,
    title="FLASK RESTful API",
    version="1.0",
    description="a boilerplate for flask restplus (restx) web service",
    authorizations=authorizations,
    security="apikey",
)

api.add_namespace(auth_ns)
api.add_namespace(user_ns, path="/user")
api.add_namespace(upload_ns, path="/upload")
api.add_namespace(train_ns, path="/train")

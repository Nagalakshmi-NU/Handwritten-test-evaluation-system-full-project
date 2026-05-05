import os
from flask import Flask, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_cors import CORS

db = SQLAlchemy()
login_manager = LoginManager()

FRONTEND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "frontend")
)

def create_app():
    app = Flask(__name__, static_folder=os.path.join(FRONTEND_DIR, "static"), static_url_path="/static")
    app.config.from_object("config.Config")

    CORS(app, supports_credentials=True)
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"

    from app.routes.auth import auth
    from app.routes.teacher import teacher
    from app.routes.student import student
    from app.routes.admin import admin
    app.register_blueprint(auth)
    app.register_blueprint(teacher)
    app.register_blueprint(student)
    app.register_blueprint(admin)

    @app.route("/")
    def index():
        return send_from_directory(os.path.join(FRONTEND_DIR, "static", "pages"), "index.html")

    return app
from flask import Flask
from flask_cors import CORS

if __name__ == "__main__":
    app = Flask(__name__)
    CORS(app)

    # Register Blueprints
    from routes.main_routes import main_bp
    from routes.predict_routes import predict_bp
    from routes.test_routes import test_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(test_bp)    
    app.run(debug=True, port=5000)

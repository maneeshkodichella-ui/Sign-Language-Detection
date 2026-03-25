import os
from dotenv import load_dotenv
from flask import Flask, request, redirect, url_for
from flask_login import LoginManager, current_user
from flask_socketio import SocketIO
from datetime import datetime
from db_setup import db, migrate
from models.user_model import User
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

socketio: Optional[SocketIO] = None


def create_app() -> Flask:
    global socketio

    load_dotenv()

    app = Flask(__name__, template_folder="templates", static_folder="static")

    # ── Security ──────────────────────────────────────────────────
    app.secret_key = os.getenv("SECRET_KEY", os.urandom(24).hex())
    app.config['SESSION_COOKIE_SECURE'] = os.getenv('FLASK_ENV') == 'production'
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    app.config['PERMANENT_SESSION_LIFETIME'] = 3600

    # ── Database ─────────────────────────────────────────────────
    # Always use SQLite (works on free cloud platforms with no extra setup).
    # To switch to MySQL set DATABASE_URL env var to a mysql+pymysql:// URI.
    database_url = os.getenv("DATABASE_URL", "sqlite:///database.db")

    # Render free tier uses /tmp for writable storage; keep SQLite there.
    if database_url.startswith("sqlite:///") and os.getenv("FLASK_ENV") == "production":
        db_path = "/tmp/database.db"
        database_url = f"sqlite:///{db_path}"
        logger.info(f"Production SQLite path: {db_path}")

    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # SQLite doesn't use connection pooling the same way MySQL does,
    # so only add pool options for non-SQLite databases.
    if not database_url.startswith("sqlite"):
        app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
            "pool_pre_ping": True,
            "pool_recycle": 3600,
            "pool_size": 5,
            "max_overflow": 10,
            "pool_timeout": 30,
        }

    # ── Extensions ────────────────────────────────────────────────
    try:
        db.init_app(app)
        migrate.init_app(app, db)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode='eventlet',   # eventlet works better on Render than threading
        logger=False,            # reduce noise in production logs
        engineio_logger=False,
        ping_timeout=60,
        ping_interval=25
    )
    logger.info("Socket.IO initialized successfully")

    # ── Login Manager ─────────────────────────────────────────────
    login_manager = LoginManager()
    login_manager.login_view = "auth.login"  # type: ignore[assignment]
    login_manager.login_message = "Please log in to access this page."
    login_manager.login_message_category = "info"
    login_manager.session_protection = "strong"
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id: str) -> Optional[User]:
        try:
            return User.query.get(int(user_id))
        except Exception as e:
            logger.error(f"Error loading user {user_id}: {e}")
            return None

    @login_manager.unauthorized_handler
    def unauthorized():
        if request.endpoint and _is_guest_allowed_route(request.endpoint):
            return None
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return {"error": "Unauthorized"}, 401
        return redirect(url_for('auth.login', next=request.url))

    def _is_guest_allowed_route(endpoint: str) -> bool:
        guest_allowed = [
            'main.index', 'main.home',
            'detection.detect', 'detection.practice',
            'detection.detection_state', 'detection.clear_word',
            'detection.delete_letter', 'detection.stop_camera',
            'detection.stop', 'detection.detection_status',
            'static', 'health_check',
        ]
        return endpoint in guest_allowed

    @app.before_request
    def _update_last_seen():
        if current_user.is_authenticated:
            try:
                current_user.last_seen = datetime.utcnow()
                db.session.commit()
            except Exception as e:
                logger.error(f"Error updating last_seen: {e}")
                db.session.rollback()

    @app.before_request
    def _check_db_connection():
        try:
            db.session.execute(db.text("SELECT 1"))
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            db.session.rollback()
            try:
                db.session.remove()
            except Exception:
                pass

    # ── Error handlers ────────────────────────────────────────────
    @app.errorhandler(404)
    def not_found_error(error):
        logger.warning(f"404: {request.url}")
        return {"error": "Page not found"}, 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"500: {error}")
        db.session.rollback()
        return {"error": "Internal server error"}, 500

    @app.errorhandler(Exception)
    def handle_exception(e):
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        db.session.rollback()
        return {"error": "An unexpected error occurred"}, 500

    # ── Health check ──────────────────────────────────────────────
    @app.route('/health')
    def health_check():
        try:
            db.session.execute(db.text("SELECT 1"))
            db_status = "healthy"
        except Exception as e:
            logger.error(f"DB health check failed: {e}")
            db_status = "unhealthy"
        return {
            "status": "healthy" if db_status == "healthy" else "degraded",
            "database": db_status,
            "timestamp": datetime.utcnow().isoformat()
        }

    # ── Blueprints ────────────────────────────────────────────────
    try:
        from routes.auth_routes import auth_bp
        from routes.main_routes import main_bp
        from routes.detection_routes import detection_bp
        from routes.admin_routes import admin_bp

        app.register_blueprint(main_bp)
        app.register_blueprint(auth_bp, url_prefix="/auth")
        app.register_blueprint(detection_bp)
        app.register_blueprint(admin_bp)
        logger.info("All blueprints registered")
    except Exception as e:
        logger.error(f"Failed to register blueprints: {e}")
        raise

    from routes.detection_routes import register_socketio_events
    register_socketio_events(socketio)
    logger.info("Socket.IO events registered")

    with app.app_context():
        try:
            db.create_all()
            logger.info("Database tables created/verified")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")

    setattr(app, 'socketio', socketio)
    return app


app = create_app()

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV", "development") == "development"

    logger.info(f"Starting on {host}:{port} (debug={debug})")

    if socketio is None:
        raise RuntimeError("Socket.IO instance is None")

    socketio.run(app, host=host, port=port, debug=debug, use_reloader=debug, log_output=True)
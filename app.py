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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global socketio instance for easy access
socketio: Optional[SocketIO] = None


def create_app() -> Flask:
    """Application factory pattern for creating Flask app"""
    global socketio
    
    load_dotenv()

    app = Flask(__name__, template_folder="templates", static_folder="static")
    
    # Security configuration
    app.secret_key = os.getenv("SECRET_KEY", os.urandom(24).hex())
    
    # Session configuration for better security
    app.config['SESSION_COOKIE_SECURE'] = os.getenv('FLASK_ENV') == 'production'
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour

    # Database configuration via SQLAlchemy + PyMySQL
    mysql_host = os.getenv("MYSQL_HOST", "localhost")
    mysql_user = os.getenv("MYSQL_USER", "root")
    mysql_password = os.getenv("MYSQL_PASSWORD", "")
    mysql_db = os.getenv("MYSQL_DB", "sign_language_db")
    mysql_port = os.getenv("MYSQL_PORT", "3306")

    # Build database URI - handle empty password correctly
    if mysql_password:
        db_uri = f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_db}"
    else:
        # If password is empty, don't include the colon
        db_uri = f"mysql+pymysql://{mysql_user}@{mysql_host}:{mysql_port}/{mysql_db}"
    
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    
    # Enhanced connection pool settings
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_pre_ping": True,  # Verify connections before using
        "pool_recycle": 3600,   # Recycle connections after 1 hour
        "pool_size": 10,        # Maximum number of connections
        "max_overflow": 20,     # Additional connections allowed
        "pool_timeout": 30,     # Timeout for getting connection from pool
        "connect_args": {
            "connect_timeout": 10  # MySQL connection timeout
        }
    }

    # Initialize extensions
    try:
        db.init_app(app)
        migrate.init_app(app, db)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

    # Initialize Socket.IO
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode='threading',
        logger=True,
        engineio_logger=True,
        ping_timeout=60,
        ping_interval=25
    )
    logger.info("Socket.IO initialized successfully")

    # Login Manager configuration
    login_manager = LoginManager()
    login_manager.login_view = "auth.login"  # type: ignore[assignment]
    login_manager.login_message = "Please log in to access this page."
    login_manager.login_message_category = "info"
    login_manager.session_protection = "strong"  # Enhanced session protection
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id: str) -> Optional[User]:
        """Load user by ID for Flask-Login"""
        try:
            return User.query.get(int(user_id))
        except Exception as e:
            logger.error(f"Error loading user {user_id}: {e}")
            return None

    @login_manager.unauthorized_handler
    def unauthorized():
        """Handle unauthorized access attempts"""
        # Check if the route allows guest access
        if request.endpoint and _is_guest_allowed_route(request.endpoint):
            # Allow access to continue without login
            return None
        
        # If request is AJAX, return 401
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return {"error": "Unauthorized"}, 401
        # Otherwise redirect to login
        return redirect(url_for('auth.login', next=request.url))

    def _is_guest_allowed_route(endpoint: str) -> bool:
        """Check if route allows guest access"""
        # Routes that allow guest access (customize as needed)
        guest_allowed = [
            'main.index',
            'main.home',
            'detection.detect',  # Sign language detection page
            'detection.practice',  # Practice page
            'detection.detection_state',  # Detection state
            'detection.clear_word',  # Clear current word
            'detection.delete_letter',  # Delete letter
            'detection.stop_camera',  # Stop camera
            'detection.stop',  # Stop camera (alternate route)
            'detection.detection_status',  # Detection status
            'static',
            'health_check',
        ]
        return endpoint in guest_allowed

    # Update last_seen on each request for authenticated users
    @app.before_request
    def _update_last_seen():
        """Update user's last_seen timestamp"""
        if current_user.is_authenticated:
            try:
                current_user.last_seen = datetime.utcnow()
                db.session.commit()
            except Exception as e:
                logger.error(f"Error updating last_seen: {e}")
                db.session.rollback()

    # Database connection error handling
    @app.before_request
    def _check_db_connection():
        """Verify database connection is alive"""
        try:
            # Ping database to ensure connection is active
            db.session.execute(db.text("SELECT 1"))
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            db.session.rollback()
            # Attempt to reconnect
            try:
                db.session.remove()
            except Exception:
                pass

    # Error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        """Handle 404 errors"""
        logger.warning(f"404 error: {request.url}")
        return {"error": "Page not found"}, 404

    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors"""
        logger.error(f"500 error: {error}")
        db.session.rollback()
        return {"error": "Internal server error"}, 500

    @app.errorhandler(Exception)
    def handle_exception(e):
        """Handle uncaught exceptions"""
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        db.session.rollback()
        return {"error": "An unexpected error occurred"}, 500

    # Health check endpoint
    @app.route('/health')
    def health_check():
        """Health check endpoint for monitoring"""
        try:
            # Check database connection
            db.session.execute(db.text("SELECT 1"))
            db_status = "healthy"
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            db_status = "unhealthy"
        
        return {
            "status": "healthy" if db_status == "healthy" else "degraded",
            "database": db_status,
            "timestamp": datetime.utcnow().isoformat()
        }

    # Register blueprints
    try:
        from routes.auth_routes import auth_bp
        from routes.main_routes import main_bp
        from routes.detection_routes import detection_bp
        from routes.admin_routes import admin_bp

        app.register_blueprint(main_bp)
        app.register_blueprint(auth_bp, url_prefix="/auth")
        app.register_blueprint(detection_bp)
        app.register_blueprint(admin_bp)
        
        logger.info("All blueprints registered successfully")
    except Exception as e:
        logger.error(f"Failed to register blueprints: {e}")
        raise

    # Register Socket.IO events after blueprints
    from routes.detection_routes import register_socketio_events
    register_socketio_events(socketio)
    logger.info("Socket.IO events registered successfully")

    # Create tables if they don't exist
    with app.app_context():
        try:
            db.create_all()
            logger.info("Database tables created/verified")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")

    # Store socketio instance in app for access in routes
    # Use setattr to avoid type checker warnings about dynamic attributes
    setattr(app, 'socketio', socketio)

    return app


# Create the app instance for Flask CLI
app = create_app()


if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV", "development") == "development"
    
    logger.info(f"Starting Flask app with Socket.IO on {host}:{port} (debug={debug})")
    
    # Ensure socketio is initialized before running
    if socketio is None:
        logger.error("Socket.IO not initialized!")
        raise RuntimeError("Socket.IO instance is None")
    
    # Use socketio.run instead of app.run
    socketio.run(
        app,
        host=host,
        port=port,
        debug=debug,
        use_reloader=debug,
        log_output=True
    )
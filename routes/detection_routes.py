from flask import Blueprint, render_template, jsonify, request, session
from flask_login import current_user, login_required
from flask_socketio import emit, join_room, leave_room
from datetime import datetime, timedelta
import logging
import atexit
import time
import gc
from sqlalchemy import func

from detector import VideoCamera, get_latest_label, get_detection_count
from models.detection_model import Detection
from db_setup import db

logger = logging.getLogger(__name__)

detection_bp = Blueprint('detection', __name__)

# Store active camera instances per user/session
active_cameras = {}


def get_user_identifier():
    """Get user identifier - either user ID or session ID for guests"""
    if current_user.is_authenticated:
        return f"user_{current_user.id}"
    else:
        if 'guest_id' not in session:
            session['guest_id'] = f"guest_{int(time.time() * 1000)}"
        return session['guest_id']


def cleanup_camera(identifier):
    """Cleanup specific camera instance"""
    global active_cameras
    
    if identifier in active_cameras:
        try:
            logger.info(f"🧹 Cleaning up camera for {identifier}")
            camera = active_cameras[identifier]
            camera.stop()
            time.sleep(0.3)
            del active_cameras[identifier]
            gc.collect()
            logger.info(f"✅ Camera cleaned up for {identifier}")
            return True
        except Exception as e:
            logger.error(f"❌ Error cleaning up camera for {identifier}: {e}")
            if identifier in active_cameras:
                del active_cameras[identifier]
            return False
    return True


@detection_bp.route('/practice')
def practice():
    """Practice page for ASL detection - accessible to all"""
    try:
        return render_template('practice.html', user=current_user if current_user.is_authenticated else None)
    except Exception as e:
        logger.error(f"Error rendering practice page: {e}")
        return jsonify({"error": "Failed to load practice page"}), 500


@detection_bp.route('/')
@detection_bp.route('/detect')
def detect():
    """Main detection page - accessible to all"""
    return render_template('home.html')


# Socket.IO event handlers
def register_socketio_events(socketio):
    """Register Socket.IO events for real-time video streaming"""
    
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection"""
        identifier = get_user_identifier()
        logger.info(f"🔌 Client connected: {identifier}")
        emit('connection_response', {'status': 'connected', 'identifier': identifier})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection"""
        identifier = get_user_identifier()
        logger.info(f"🔌 Client disconnected: {identifier}")
        cleanup_camera(identifier)
    
    @socketio.on('start_detection')
    def handle_start_detection():
        """Start video detection for this client"""
        try:
            identifier = get_user_identifier()
            logger.info(f"▶️ Starting detection for {identifier}")
            
            # Clean up any existing camera
            cleanup_camera(identifier)
            
            # Create room for this user
            join_room(identifier)
            
            # Create new camera with Socket.IO
            camera = VideoCamera(socketio=socketio, room=identifier)
            active_cameras[identifier] = camera
            
            # Start detection loop
            camera.start_detection()
            
            emit('detection_started', {
                'status': 'success',
                'message': 'Detection started successfully'
            })
            
        except Exception as e:
            logger.error(f"❌ Error starting detection: {e}", exc_info=True)
            emit('error', {'message': f'Failed to start detection: {str(e)}'})
    
    @socketio.on('stop_detection')
    def handle_stop_detection():
        """Stop video detection for this client"""
        try:
            identifier = get_user_identifier()
            logger.info(f"⏹️ Stopping detection for {identifier}")
            
            cleanup_camera(identifier)
            leave_room(identifier)
            
            emit('detection_stopped', {
                'status': 'success',
                'message': 'Detection stopped successfully'
            })
            
        except Exception as e:
            logger.error(f"❌ Error stopping detection: {e}")
            emit('error', {'message': f'Failed to stop detection: {str(e)}'})
    
    @socketio.on('reset_count')
    def handle_reset_count():
        """Reset detection counter"""
        try:
            identifier = get_user_identifier()
            if identifier in active_cameras:
                active_cameras[identifier].reset_count()
                emit('count_reset', {'status': 'success'})
            else:
                emit('error', {'message': 'Camera not initialized'})
        except Exception as e:
            logger.error(f"❌ Error resetting count: {e}")
            emit('error', {'message': str(e)})
    
    @socketio.on('get_detection_state')
    def handle_get_detection_state():
        """Get current detection state"""
        try:
            label, confidence = get_latest_label()
            count = get_detection_count()
            
            # Save detection to database only for authenticated users
            if current_user.is_authenticated and label and confidence > 0.75:
                try:
                    recent_time = datetime.utcnow() - timedelta(seconds=2)
                    recent_detection = Detection.query.filter(
                        Detection.user_id == current_user.id,
                        Detection.detected_sign == label,
                        Detection.timestamp >= recent_time
                    ).first()
                    
                    if not recent_detection:
                        detection = Detection()
                        detection.user_id = current_user.id
                        detection.detected_sign = label
                        db.session.add(detection)
                        db.session.commit()
                except Exception as e:
                    logger.error(f"❌ Error saving detection: {e}")
                    db.session.rollback()
            
            emit('detection_state', {
                'label': label,
                'confidence': float(confidence) if confidence else 0.0,
                'detection_count': count
            })
            
        except Exception as e:
            logger.error(f"❌ Error getting detection state: {e}")
            emit('error', {'message': str(e)})


# Legacy HTTP endpoints (for backward compatibility)
@detection_bp.route('/detection_state')
def detection_state():
    """Get complete detection state - HTTP fallback"""
    try:
        label, confidence = get_latest_label()
        count = get_detection_count()
        
        if current_user.is_authenticated and label and confidence > 0.75:
            try:
                recent_time = datetime.utcnow() - timedelta(seconds=2)
                recent_detection = Detection.query.filter(
                    Detection.user_id == current_user.id,
                    Detection.detected_sign == label,
                    Detection.timestamp >= recent_time
                ).first()
                
                if not recent_detection:
                    detection = Detection()
                    detection.user_id = current_user.id
                    detection.detected_sign = label
                    db.session.add(detection)
                    db.session.commit()
            except Exception as e:
                logger.error(f"❌ Error saving detection: {e}")
                db.session.rollback()
        
        return jsonify({
            "label": label,
            "confidence": float(confidence) if confidence else 0.0,
            "detection_count": count
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting detection state: {e}", exc_info=True)
        return jsonify({
            "label": "",
            "confidence": 0.0,
            "detection_count": 0
        }), 200


@detection_bp.route('/reset_count', methods=['POST'])
def reset_count():
    """Reset detection counter"""
    try:
        identifier = get_user_identifier()
        if identifier in active_cameras:
            active_cameras[identifier].reset_count()
            return jsonify({"success": True, "message": "Counter reset"})
        return jsonify({"success": False, "message": "Camera not initialized"}), 400
    except Exception as e:
        logger.error(f"Error resetting count: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@detection_bp.route('/stop')
@detection_bp.route('/stop_camera', methods=['GET', 'POST'])
def stop_camera():
    """Stop and release camera for current user/guest"""
    try:
        identifier = get_user_identifier()
        
        if cleanup_camera(identifier):
            return jsonify({
                "success": True,
                "status": "Camera stopped",
                "message": "Camera stopped successfully"
            })
        else:
            return jsonify({
                "success": False,
                "status": "Camera cleanup had errors",
                "message": "Camera stopped with errors"
            }), 500
        
    except Exception as e:
        logger.error(f"❌ Error in stop_camera: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@detection_bp.route('/detection_status')
def detection_status():
    """Get detection system status"""
    try:
        identifier = get_user_identifier()
        is_active = identifier in active_cameras and active_cameras[identifier].is_running
        
        label, conf = get_latest_label()
        count = get_detection_count()
        
        return jsonify({
            "active": is_active,
            "current_label": label,
            "confidence": float(conf),
            "detection_count": count
        })
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({"active": False, "error": str(e)}), 500


# Authenticated-only routes (history/stats)
@detection_bp.route('/detection_history')
@login_required
def detection_history():
    """Get user's detection history"""
    try:
        detections = Detection.query.filter_by(
            user_id=current_user.id
        ).order_by(
            Detection.timestamp.desc()
        ).limit(100).all()
        
        history = [
            {
                "id": d.id,
                "sign": d.detected_sign,
                "timestamp": d.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            }
            for d in detections
        ]
        
        return jsonify({
            "success": True,
            "history": history
        })
    except Exception as e:
        logger.error(f"❌ Error fetching detection history: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@detection_bp.route('/clear_history', methods=["POST"])
@login_required
def clear_history():
    """Clear user's detection history"""
    try:
        deleted_count = Detection.query.filter_by(
            user_id=current_user.id
        ).delete()
        
        db.session.commit()
        logger.info(f"🗑️ Cleared {deleted_count} detections for user {current_user.id}")
        
        return jsonify({
            "success": True,
            "message": f"Cleared {deleted_count} detection(s) from history"
        })
    except Exception as e:
        logger.error(f"❌ Error clearing history: {e}")
        db.session.rollback()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@detection_bp.route('/detection_stats')
@login_required
def detection_stats():
    """Get user's detection statistics"""
    try:
        total_detections = Detection.query.filter_by(
            user_id=current_user.id
        ).count()
        
        unique_signs = db.session.query(
            Detection.detected_sign
        ).filter_by(
            user_id=current_user.id
        ).distinct().count()
        
        recent_detection = Detection.query.filter_by(
            user_id=current_user.id
        ).order_by(
            Detection.timestamp.desc()
        ).first()
        
        most_common = db.session.query(
            Detection.detected_sign,
            func.count(Detection.id).label('count')
        ).filter_by(
            user_id=current_user.id
        ).group_by(
            Detection.detected_sign
        ).order_by(
            func.count(Detection.id).desc()
        ).first()
        
        return jsonify({
            "success": True,
            "stats": {
                "total_detections": total_detections,
                "unique_signs": unique_signs,
                "last_detection": recent_detection.timestamp.strftime('%Y-%m-%d %H:%M:%S') if recent_detection else None,
                "most_common_sign": most_common[0] if most_common else None,
                "most_common_count": most_common[1] if most_common else 0
            }
        })
    except Exception as e:
        logger.error(f"❌ Error fetching stats: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# Cleanup functions
def cleanup_all_cameras():
    """Clean up all active cameras"""
    global active_cameras
    logger.info("🧹 Cleaning up all cameras...")
    
    for identifier in list(active_cameras.keys()):
        cleanup_camera(identifier)
    
    active_cameras.clear()
    gc.collect()
    logger.info("✅ All cameras cleaned up")


# Register cleanup on exit
atexit.register(cleanup_all_cameras)


@detection_bp.teardown_app_request
def cleanup_camera_on_error(exception=None):
    """Cleanup camera on request end if there was an error"""
    if exception:
        try:
            identifier = get_user_identifier()
            cleanup_camera(identifier)
            logger.info(f"🧹 Camera cleaned up after error for {identifier}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
from flask import Blueprint, render_template, abort
from flask_login import login_required, current_user
from models.user_model import User
from models.detection_model import Detection
from datetime import datetime


admin_bp = Blueprint("admin", __name__)


@admin_bp.route("/admin")
@login_required
def admin_home():
    if not getattr(current_user, "is_admin", False):
        abort(403)
    users = User.query.order_by(User.created_at.desc()).all()
    detections = Detection.query.order_by(Detection.timestamp.desc()).limit(500).all()
    return render_template("admin.html", users=users, detections=detections, now=datetime.utcnow())



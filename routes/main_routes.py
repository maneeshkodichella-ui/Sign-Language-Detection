from flask import Blueprint, render_template
from flask_login import login_required, current_user


main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def home():
    """Home page"""
    return render_template("home.html")


@main_bp.route("/about")
def about():
    """About page"""
    return render_template("about.html")


@main_bp.route("/learn")
def learn():
    """Learn ASL gestures page"""
    return render_template("learn_gestures.html")


@main_bp.route("/profile")
@login_required
def profile():
    """User profile page"""
    return render_template("profile.html", user=current_user)


@main_bp.route("/detect")
@login_required
def detect():
    """Sign language detection page"""
    return render_template("detect.html", user=current_user)


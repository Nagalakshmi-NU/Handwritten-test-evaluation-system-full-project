from flask import Blueprint, request, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from app import db
from app.models import User, ActivityLog, Notification

auth = Blueprint("auth", __name__)

def log(user_id, action):
    db.session.add(ActivityLog(user_id=user_id, action=action))
    db.session.commit()

@auth.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    if not data.get("email") or not data.get("password") or not data.get("name"):
        return jsonify({"error": "Name, email and password are required"}), 400
    if User.query.filter_by(email=data["email"]).first():
        return jsonify({"error": "Email already exists"}), 400
    role = data.get("role", "student")
    if role == "admin":
        return jsonify({"error": "Cannot self-register as admin"}), 403
    user = User(
        name=data["name"], email=data["email"],
        password_hash=generate_password_hash(data["password"]),
        role=role,
        employee_id=data.get("employee_id"),
        student_id=data.get("student_id"),
        subjects=data.get("subjects"),
        status="pending"
    )
    db.session.add(user)
    db.session.commit()
    log(user.id, f"Registered as {role} — pending approval")
    return jsonify({"message": "Registration request sent. Waiting for admin approval."}), 201

@auth.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    user = User.query.filter_by(email=data.get("email")).first()
    if not user:
        return jsonify({"error": "Invalid credentials"}), 401
    # Check if account is locked
    if user.locked_until and datetime.utcnow() < user.locked_until:
        remaining = int((user.locked_until - datetime.utcnow()).total_seconds() / 60) + 1
        return jsonify({"error": f"Account locked. Try again in {remaining} minute(s)."}), 403
    if not check_password_hash(user.password_hash, data.get("password", "")):
        user.login_attempts = (user.login_attempts or 0) + 1
        if user.login_attempts >= 5:
            from datetime import timedelta
            user.locked_until = datetime.utcnow() + timedelta(minutes=15)
            user.login_attempts = 0
            db.session.commit()
            return jsonify({"error": "Too many failed attempts. Account locked for 15 minutes."}), 403
        db.session.commit()
        remaining = 5 - user.login_attempts
        return jsonify({"error": f"Invalid credentials. {remaining} attempt(s) remaining."}), 401
    if user.status == "pending":
        return jsonify({"error": "Your account is pending admin approval."}), 403
    if user.status == "blocked":
        return jsonify({"error": "Your account has been blocked. Contact admin."}), 403
    # Reset attempts on success
    user.login_attempts = 0
    user.locked_until   = None
    user.last_login     = datetime.utcnow()
    db.session.commit()
    login_user(user)
    log(user.id, "Logged in")
    return jsonify({"message": "Logged in", "role": user.role, "name": user.name}), 200

@auth.route("/logout", methods=["POST"])
@login_required
def logout():
    log(current_user.id, "Logged out")
    logout_user()
    return jsonify({"message": "Logged out"}), 200

@auth.route("/me", methods=["GET"])
@login_required
def me():
    return jsonify({
        "id": current_user.id, "name": current_user.name,
        "email": current_user.email, "role": current_user.role,
        "status": current_user.status,
        "initials": ''.join([w[0].upper() for w in current_user.name.split()[:2]]),
        "employee_id": current_user.employee_id,
        "student_id": current_user.student_id,
        "admin_id": current_user.admin_id
    }), 200

@auth.route("/me/update", methods=["POST"])
@login_required
def update_profile():
    data = request.get_json()
    if "name" in data and data["name"].strip():
        current_user.name = data["name"].strip()
    if "new_password" in data and data["new_password"]:
        old = data.get("old_password", "")
        if not check_password_hash(current_user.password_hash, old):
            return jsonify({"error": "Current password is incorrect"}), 400
        if len(data["new_password"]) < 6:
            return jsonify({"error": "Password must be at least 6 characters"}), 400
        current_user.password_hash = generate_password_hash(data["new_password"])
    db.session.commit()
    log(current_user.id, "Updated profile")
    return jsonify({"message": "Profile updated",
                    "name": current_user.name,
                    "initials": ''.join([w[0].upper() for w in current_user.name.split()[:2]])}), 200

@auth.route("/forgot-password", methods=["POST"])
def forgot_password():
    data  = request.get_json()
    email = data.get("email", "").strip()
    if not email:
        return jsonify({"error": "Email is required"}), 400
    user = User.query.filter_by(email=email).first()
    if not user:
        # Don't reveal if email exists
        return jsonify({"message": "If this email exists, a reset code has been sent."}), 200
    # Generate 6-digit reset code
    import random, string
    code = ''.join(random.choices(string.digits, k=6))
    # Store code in DB temporarily (reuse admin_id field as temp storage)
    user.admin_id = f"RESET:{code}:{int(datetime.utcnow().timestamp())}"
    db.session.commit()
    log(user.id, "Requested password reset")
    # In production: send email. For now return code directly (demo only)
    return jsonify({"message": "Reset code generated.", "code": code, "demo": True}), 200

@auth.route("/reset-password", methods=["POST"])
def reset_password():
    data     = request.get_json()
    email    = data.get("email", "").strip()
    code     = data.get("code", "").strip()
    new_pass = data.get("new_password", "")
    if not email or not code or not new_pass:
        return jsonify({"error": "Email, code and new password are required"}), 400
    if len(new_pass) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    user = User.query.filter_by(email=email).first()
    if not user or not user.admin_id or not user.admin_id.startswith("RESET:"):
        return jsonify({"error": "Invalid or expired reset code"}), 400
    parts = user.admin_id.split(":")
    stored_code = parts[1]
    timestamp   = int(parts[2])
    # Code expires in 10 minutes
    if int(datetime.utcnow().timestamp()) - timestamp > 600:
        return jsonify({"error": "Reset code has expired. Please request a new one."}), 400
    if code != stored_code:
        return jsonify({"error": "Invalid reset code"}), 400
    user.password_hash = generate_password_hash(new_pass)
    user.admin_id = None
    db.session.commit()
    log(user.id, "Password reset successfully")
    return jsonify({"message": "Password reset successfully. You can now login."}), 200

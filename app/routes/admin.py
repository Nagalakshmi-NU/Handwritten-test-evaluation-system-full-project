from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from werkzeug.security import generate_password_hash
from app import db
from app.models import User, Test, Submission, Evaluation, Question, ActivityLog, Notification
from functools import wraps
import csv, io

admin = Blueprint("admin", __name__)

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != "admin":
            return jsonify({"error": "Admin access required"}), 403
        return f(*args, **kwargs)
    return decorated

def log(user_id, action):
    db.session.add(ActivityLog(user_id=user_id, action=action))
    db.session.commit()

# ── STATS ───────────────────────────────────────────────────
@admin.route("/admin/stats", methods=["GET"])
@login_required
@admin_required
def get_stats():
    subs = Submission.query.all()
    status_counts = {}
    for s in subs:
        status_counts[s.status] = status_counts.get(s.status, 0) + 1
    return jsonify({
        "total_teachers":    User.query.filter_by(role="teacher").count(),
        "total_students":    User.query.filter_by(role="student").count(),
        "pending_requests":  User.query.filter_by(status="pending").count(),
        "blocked_users":     User.query.filter_by(status="blocked").count(),
        "total_tests":       Test.query.count(),
        "active_tests":      Test.query.filter_by(status="active").count(),
        "total_submissions": len(subs),
        "published":         status_counts.get("published", 0),
        "pending_subs":      status_counts.get("pending", 0),
        "status_counts":     status_counts
    }), 200

# ── ANALYTICS ───────────────────────────────────────────────
@admin.route("/admin/analytics", methods=["GET"])
@login_required
@admin_required
def get_analytics():
    # Top 5 students by avg score
    students = User.query.filter_by(role="student", status="approved").all()
    top_students = []
    for s in students:
        subs = Submission.query.filter_by(student_id=s.id, status="published").all()
        if not subs: continue
        scores = []
        for sub in subs:
            test = Test.query.get(sub.test_id)
            evals = Evaluation.query.filter_by(submission_id=sub.id).all()
            total = sum(e.final_score or 0 for e in evals)
            if test and test.total_marks:
                scores.append(round((total/test.total_marks)*100, 1))
        if scores:
            top_students.append({"name": s.name, "avg": round(sum(scores)/len(scores), 1), "tests": len(scores)})
    top_students.sort(key=lambda x: x["avg"], reverse=True)

    # Most active teachers
    teachers = User.query.filter_by(role="teacher", status="approved").all()
    teacher_activity = []
    for t in teachers:
        test_count = Test.query.filter_by(teacher_id=t.id).count()
        teacher_activity.append({"name": t.name, "tests": test_count,
                                  "last_login": t.last_login.isoformat() if t.last_login else None})
    teacher_activity.sort(key=lambda x: x["tests"], reverse=True)

    # Submission trend (last 7 days)
    from datetime import datetime, timedelta
    trend = []
    for i in range(6, -1, -1):
        day = datetime.utcnow() - timedelta(days=i)
        count = Submission.query.filter(
            db.func.date(Submission.uploaded_at) == day.date()
        ).count()
        trend.append({"date": day.strftime("%b %d"), "count": count})

    return jsonify({
        "top_students": top_students[:5],
        "teacher_activity": teacher_activity[:5],
        "submission_trend": trend
    }), 200

# ── USERS ───────────────────────────────────────────────────
@admin.route("/admin/users", methods=["GET"])
@login_required
@admin_required
def get_users():
    role   = request.args.get("role")
    status = request.args.get("status")
    search = request.args.get("search", "").strip().lower()
    q = User.query.filter(User.role != "admin")
    if role:   q = q.filter_by(role=role)
    if status: q = q.filter_by(status=status)
    users = q.all()
    if search:
        users = [u for u in users if search in u.name.lower() or search in u.email.lower()
                 or search in (u.student_id or "").lower() or search in (u.employee_id or "").lower()]
    return jsonify({"users": [_user_dict(u) for u in users]}), 200

@admin.route("/admin/users", methods=["POST"])
@login_required
@admin_required
def create_user():
    data = request.get_json()
    if User.query.filter_by(email=data["email"]).first():
        return jsonify({"error": "Email already exists"}), 400
    user = User(
        name=data["name"], email=data["email"],
        password_hash=generate_password_hash(data["password"]),
        role=data["role"], status="approved",
        employee_id=data.get("employee_id"),
        student_id=data.get("student_id"),
        subjects=data.get("subjects")
    )
    db.session.add(user)
    db.session.commit()
    log(current_user.id, f"Created user {user.email} ({user.role})")
    return jsonify({"message": "User created", "user": _user_dict(user)}), 201

@admin.route("/admin/users/<int:user_id>", methods=["PUT"])
@login_required
@admin_required
def edit_user(user_id):
    user = User.query.get_or_404(user_id)
    data = request.get_json()
    if "name" in data: user.name = data["name"]
    if "email" in data: user.email = data["email"]
    if "employee_id" in data: user.employee_id = data["employee_id"]
    if "student_id" in data: user.student_id = data["student_id"]
    if "subjects" in data: user.subjects = data["subjects"]
    db.session.commit()
    log(current_user.id, f"Edited user {user.email}")
    return jsonify({"message": "User updated", "user": _user_dict(user)}), 200

@admin.route("/admin/users/<int:user_id>/approve", methods=["POST"])
@login_required
@admin_required
def approve_user(user_id):
    user = User.query.get_or_404(user_id)
    user.status = "approved"
    db.session.add(Notification(user_id=user.id, message="✅ Your account has been approved! You can now login."))
    db.session.commit()
    log(current_user.id, f"Approved user {user.email}")
    return jsonify({"message": f"{user.name} approved"}), 200

@admin.route("/admin/users/<int:user_id>/reject", methods=["POST"])
@login_required
@admin_required
def reject_user(user_id):
    user = User.query.get_or_404(user_id)
    try:
        from app.models import ActivityLog, Notification
        ActivityLog.query.filter_by(user_id=user_id).delete()
        Notification.query.filter_by(user_id=user_id).delete()
        log(current_user.id, f"Rejected user {user.email}")
        db.session.delete(user)
        db.session.commit()
        return jsonify({"message": "User rejected"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Reject failed: {str(e)}"}), 500

@admin.route("/admin/users/<int:user_id>/block", methods=["POST"])
@login_required
@admin_required
def block_user(user_id):
    user = User.query.get_or_404(user_id)
    user.status = "blocked"
    db.session.commit()
    log(current_user.id, f"Blocked user {user.email}")
    return jsonify({"message": f"{user.name} blocked"}), 200

@admin.route("/admin/users/<int:user_id>/unblock", methods=["POST"])
@login_required
@admin_required
def unblock_user(user_id):
    user = User.query.get_or_404(user_id)
    user.status = "approved"
    db.session.commit()
    log(current_user.id, f"Unblocked user {user.email}")
    return jsonify({"message": f"{user.name} unblocked"}), 200

@admin.route("/admin/users/<int:user_id>", methods=["DELETE"])
@login_required
@admin_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    try:
        # Delete all related records first to avoid FK constraint errors
        from app.models import Submission, SubmissionPage, Evaluation, ActivityLog, Notification, TeacherFeedback

        # Get all submissions by this user
        subs = Submission.query.filter_by(student_id=user_id).all()
        for sub in subs:
            # Delete evaluations and pages for each submission
            Evaluation.query.filter_by(submission_id=sub.id).delete()
            SubmissionPage.query.filter_by(submission_id=sub.id).delete()
        Submission.query.filter_by(student_id=user_id).delete()

        # If teacher, delete their tests' submissions too
        if user.role == 'teacher':
            from app.models import Test, Question
            tests = Test.query.filter_by(teacher_id=user_id).all()
            for test in tests:
                test_subs = Submission.query.filter_by(test_id=test.id).all()
                for sub in test_subs:
                    Evaluation.query.filter_by(submission_id=sub.id).delete()
                    SubmissionPage.query.filter_by(submission_id=sub.id).delete()
                Submission.query.filter_by(test_id=test.id).delete()
                Question.query.filter_by(test_id=test.id).delete()
            Test.query.filter_by(teacher_id=user_id).delete()

        # Delete other related records
        ActivityLog.query.filter_by(user_id=user_id).delete()
        Notification.query.filter_by(user_id=user_id).delete()
        TeacherFeedback.query.filter_by(student_id=user_id).delete()
        TeacherFeedback.query.filter_by(teacher_id=user_id).delete()

        log(current_user.id, f"Deleted user {user.email}")
        db.session.delete(user)
        db.session.commit()
        return jsonify({"message": f"{user.name} deleted successfully"}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Delete failed: {str(e)}"}), 500

@admin.route("/admin/users/bulk", methods=["POST"])
@login_required
@admin_required
def bulk_action():
    data = request.get_json()
    action = data.get("action")
    ids    = data.get("ids", [])
    count  = 0
    for uid in ids:
        user = User.query.get(uid)
        if not user or user.role == "admin": continue
        if action == "approve":   user.status = "approved"; count += 1
        elif action == "block":   user.status = "blocked";  count += 1
        elif action == "delete":  db.session.delete(user);  count += 1
    db.session.commit()
    log(current_user.id, f"Bulk {action} on {count} users")
    return jsonify({"message": f"{action} applied to {count} users"}), 200

# ── TESTS ───────────────────────────────────────────────────
@admin.route("/admin/users/<int:user_id>/reset-password", methods=["POST"])
@login_required
@admin_required
def reset_user_password(user_id):
    user = User.query.get_or_404(user_id)
    data = request.get_json()
    new_password = data.get("new_password", "").strip()
    if not new_password or len(new_password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    user.password_hash = generate_password_hash(new_password)
    db.session.commit()
    log(current_user.id, f"Reset password for user {user.email}")
    return jsonify({"message": f"Password reset for {user.name}"}), 200


@admin.route("/admin/tests", methods=["GET"])
@login_required
@admin_required
def get_all_tests():
    tests = Test.query.order_by(Test.created_at.desc()).all()
    result = []
    for t in tests:
        teacher = User.query.get(t.teacher_id)
        result.append({
            "test_id": t.id, "title": t.title, "subject": t.subject,
            "teacher_name": teacher.name if teacher else "Unknown",
            "teacher_id": t.teacher_id,
            "total_marks": t.total_marks, "status": t.status,
            "submissions": Submission.query.filter_by(test_id=t.id).count(),
            "created_at": t.created_at.isoformat()
        })
    return jsonify({"tests": result}), 200

@admin.route("/admin/tests/<int:test_id>/status", methods=["POST"])
@login_required
@admin_required
def toggle_test_status(test_id):
    test = Test.query.get_or_404(test_id)
    test.status = "closed" if test.status == "active" else "active"
    db.session.commit()
    log(current_user.id, f"Changed test '{test.title}' status to {test.status}")
    return jsonify({"message": f"Test {test.status}", "status": test.status}), 200

@admin.route("/admin/tests/<int:test_id>", methods=["DELETE"])
@login_required
@admin_required
def delete_test(test_id):
    test = Test.query.get_or_404(test_id)
    log(current_user.id, f"Deleted test '{test.title}'")
    db.session.delete(test)
    db.session.commit()
    return jsonify({"message": "Test deleted"}), 200

# ── SUBMISSIONS ─────────────────────────────────────────────
@admin.route("/admin/submissions", methods=["GET"])
@login_required
@admin_required
def get_all_submissions():
    subs = Submission.query.order_by(Submission.uploaded_at.desc()).all()
    result = []
    for s in subs:
        student = User.query.get(s.student_id)
        test    = Test.query.get(s.test_id)
        result.append({
            "submission_id": s.id,
            "student_name": student.name if student else "Unknown",
            "test_title": test.title if test else "Unknown",
            "status": s.status,
            "uploaded_at": s.uploaded_at.isoformat()
        })
    return jsonify({"submissions": result}), 200

# ── TEACHER DETAILS ─────────────────────────────────────────
@admin.route("/admin/teacher/<int:teacher_id>/details", methods=["GET"])
@login_required
@admin_required
def teacher_details(teacher_id):
    teacher = User.query.get_or_404(teacher_id)
    tests   = Test.query.filter_by(teacher_id=teacher_id).all()
    logs    = ActivityLog.query.filter_by(user_id=teacher_id)\
                .order_by(ActivityLog.timestamp.desc()).limit(30).all()
    all_students = User.query.filter_by(role="student", status="approved").all()

    tests_data = []
    for t in tests:
        subs = Submission.query.filter_by(test_id=t.id).all()
        submitted_ids = {s.student_id for s in subs}
        submitted_list, scored = [], []
        for sub in subs:
            stu   = User.query.get(sub.student_id)
            evals = Evaluation.query.filter_by(submission_id=sub.id).all()
            total = sum(e.final_score or 0 for e in evals)
            pct   = round((total/t.total_marks)*100,1) if t.total_marks and sub.status=="published" else None
            entry = {"student_id": sub.student_id, "name": stu.name if stu else "Unknown",
                     "student_id_no": stu.student_id if stu else "", "status": sub.status,
                     "score": total if sub.status=="published" else None,
                     "max_marks": t.total_marks, "percentage": pct,
                     "submitted_at": sub.uploaded_at.isoformat()}
            submitted_list.append(entry)
            if pct is not None: scored.append(entry)
        scored.sort(key=lambda x: x["percentage"], reverse=True)
        not_submitted = [{"student_id": s.id, "name": s.name, "student_id_no": s.student_id or ""}
                         for s in all_students if s.id not in submitted_ids]
        tests_data.append({
            "test_id": t.id, "title": t.title, "subject": t.subject or "",
            "total_marks": t.total_marks, "status": t.status,
            "created_at": t.created_at.isoformat(),
            "total_submissions": len(subs),
            "top3": scored[:3], "submitted": submitted_list, "not_submitted": not_submitted
        })

    return jsonify({
        "teacher": _user_dict(teacher), "tests": tests_data,
        "activity_logs": [{"action": l.action, "timestamp": l.timestamp.isoformat()} for l in logs]
    }), 200

# ── STUDENT DETAILS ─────────────────────────────────────────
@admin.route("/admin/student/<int:student_id>/details", methods=["GET"])
@login_required
@admin_required
def student_details(student_id):
    student   = User.query.get_or_404(student_id)
    all_tests = Test.query.all()
    logs      = ActivityLog.query.filter_by(user_id=student_id)\
                  .order_by(ActivityLog.timestamp.desc()).limit(30).all()
    taken, not_taken = [], []
    total_pct, pub_count = 0, 0
    for t in all_tests:
        sub     = Submission.query.filter_by(test_id=t.id, student_id=student_id).first()
        teacher = User.query.get(t.teacher_id)
        if sub:
            evals = Evaluation.query.filter_by(submission_id=sub.id).all()
            score = sum(e.final_score or 0 for e in evals)
            pct   = round((score/t.total_marks)*100,1) if t.total_marks and sub.status=="published" else None
            if pct is not None: total_pct += pct; pub_count += 1
            taken.append({"test_id": t.id, "title": t.title, "subject": t.subject or "",
                          "teacher_name": teacher.name if teacher else "Unknown",
                          "total_marks": t.total_marks, "score": score if sub.status=="published" else None,
                          "percentage": pct, "status": sub.status,
                          "submitted_at": sub.uploaded_at.isoformat()})
        else:
            not_taken.append({"test_id": t.id, "title": t.title, "subject": t.subject or "",
                               "teacher_name": teacher.name if teacher else "Unknown",
                               "total_marks": t.total_marks})
    avg  = round(total_pct/pub_count, 1) if pub_count else 0
    best = max((r["percentage"] for r in taken if r["percentage"] is not None), default=0)
    return jsonify({
        "student": _user_dict(student), "taken": taken, "not_taken": not_taken,
        "avg_percentage": avg, "best_score": best,
        "total_tests": len(all_tests), "tests_taken": len(taken),
        "activity_logs": [{"action": l.action, "timestamp": l.timestamp.isoformat()} for l in logs]
    }), 200

# ── BROADCAST ───────────────────────────────────────────────
@admin.route("/admin/broadcast", methods=["POST"])
@login_required
@admin_required
def broadcast():
    data    = request.get_json()
    message = data.get("message", "").strip()
    target  = data.get("target", "all")  # all / teachers / students
    if not message:
        return jsonify({"error": "Message required"}), 400
    q = User.query.filter(User.role != "admin", User.status == "approved")
    if target == "teachers": q = q.filter_by(role="teacher")
    elif target == "students": q = q.filter_by(role="student")
    users = q.all()
    for u in users:
        db.session.add(Notification(user_id=u.id, message=f"📢 Admin: {message}"))
    db.session.commit()
    log(current_user.id, f"Broadcast to {target}: {message[:50]}")
    return jsonify({"message": f"Sent to {len(users)} users"}), 200

# ── LOGS ────────────────────────────────────────────────────
@admin.route("/admin/logs", methods=["GET"])
@login_required
@admin_required
def get_logs():
    role   = request.args.get("role")
    search = request.args.get("search", "").strip().lower()
    limit  = int(request.args.get("limit", 100))
    logs   = ActivityLog.query.order_by(ActivityLog.timestamp.desc()).limit(limit).all()
    result = []
    for l in logs:
        user = User.query.get(l.user_id)
        if role and (not user or user.role != role): continue
        if search and search not in (l.action.lower() + (user.name.lower() if user else "")): continue
        result.append({
            "user_name": user.name if user else "Unknown",
            "user_role": user.role if user else "",
            "action": l.action,
            "timestamp": l.timestamp.isoformat()
        })
    return jsonify({"logs": result}), 200

@admin.route("/admin/logs/export", methods=["GET"])
@login_required
@admin_required
def export_logs():
    from flask import Response
    logs = ActivityLog.query.order_by(ActivityLog.timestamp.desc()).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Timestamp", "User", "Role", "Action"])
    for l in logs:
        user = User.query.get(l.user_id)
        writer.writerow([l.timestamp.isoformat(), user.name if user else "Unknown",
                         user.role if user else "", l.action])
    return Response(output.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment;filename=activity_logs.csv"})

@admin.route("/admin/users/export", methods=["GET"])
@login_required
@admin_required
def export_users():
    from flask import Response
    users = User.query.filter(User.role != "admin").all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Name", "Email", "Role", "ID", "Status", "Created At", "Last Login"])
    for u in users:
        uid = u.employee_id or u.student_id or "—"
        writer.writerow([u.name, u.email, u.role, uid, u.status,
                         u.created_at.isoformat(), u.last_login.isoformat() if u.last_login else "Never"])
    return Response(output.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment;filename=users.csv"})

def _user_dict(u):
    return {
        "id": u.id, "name": u.name, "email": u.email, "role": u.role,
        "status": u.status, "employee_id": u.employee_id,
        "student_id": u.student_id, "subjects": u.subjects,
        "admin_id": u.admin_id, "created_at": u.created_at.isoformat(),
        "last_login": u.last_login.isoformat() if u.last_login else None,
        "initials": ''.join([w[0].upper() for w in u.name.split()[:2]])
    }

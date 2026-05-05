from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from app import db
from app.models import Submission, SubmissionPage, Test, Evaluation, Question, ActivityLog, Notification
from app.tasks.ocr_tasks import process_submission, _run_pipeline
import os

student = Blueprint("student", __name__)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'pdf'}
MAX_FILE_SIZE_MB = 10

def allowed_file(f): return '.' in f and f.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def log(user_id, action):
    db.session.add(ActivityLog(user_id=user_id, action=action))
    db.session.commit()

@student.route("/student/tests", methods=["GET"])
@login_required
def get_student_tests():
    if current_user.role != "student":
        return jsonify({"error": "Students only"}), 403
    from datetime import datetime
    tests = Test.query.filter_by(status="active").all()
    result = []
    for t in tests:
        sub = Submission.query.filter_by(test_id=t.id, student_id=current_user.id).first()
        now = datetime.utcnow()
        deadline_passed = t.deadline and now > t.deadline
        result.append({
            "test_id": t.id, "title": t.title,
            "subject": t.subject, "total_marks": t.total_marks,
            "submission_id": sub.id if sub else None,
            "submission_status": sub.status if sub else None,
            "deadline": t.deadline.isoformat() if t.deadline else None,
            "deadline_passed": deadline_passed
        })
    return jsonify({"tests": result}), 200

@student.route("/student/test/<int:test_id>/questions", methods=["GET"])
@login_required
def get_test_questions(test_id):
    """Get questions for a test so student can view the question paper."""
    test = Test.query.get_or_404(test_id)
    questions = Question.query.filter_by(test_id=test_id).order_by(Question.question_number).all()
    return jsonify({
        "test_id": test.id,
        "title": test.title,
        "subject": test.subject or "",
        "total_marks": test.total_marks,
        "questions": [{
            "question_number": q.question_number,
            "question_text": q.question_text,
            "max_marks": q.max_marks
        } for q in questions]
    }), 200

@student.route("/submit", methods=["POST"])
@login_required
def submit():
    if current_user.role != "student":
        return jsonify({"error": "Only students can submit"}), 403

    test_id = request.form.get("test_id")
    if not test_id:
        return jsonify({"error": "test_id is required"}), 400
    try:
        test_id = int(test_id)
    except ValueError:
        return jsonify({"error": "test_id must be a number"}), 400

    test = Test.query.get(test_id)
    if not test:
        return jsonify({"error": f"Test {test_id} not found"}), 404

    # Check deadline
    from datetime import datetime
    if test.deadline and datetime.utcnow() > test.deadline:
        return jsonify({"error": "Submission deadline has passed. You can no longer upload answers for this test."}), 403

    files = request.files.getlist("image")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No image file uploaded"}), 400

    for f in files:
        if not allowed_file(f.filename):
            return jsonify({"error": f"Invalid file type: {f.filename}. Allowed: jpg, jpeg, png, pdf"}), 400
        f.seek(0, 2)
        if f.tell() / (1024 * 1024) > MAX_FILE_SIZE_MB:
            return jsonify({"error": f"File too large. Max {MAX_FILE_SIZE_MB}MB"}), 400
        f.seek(0)

    existing = Submission.query.filter_by(test_id=test_id, student_id=current_user.id).first()
    if existing:
        return jsonify({"error": "Already submitted", "existing_submission_id": existing.id, "status": existing.status}), 400

    try:
        upload_dir = os.path.join(BASE_DIR, "uploads")
        os.makedirs(upload_dir, exist_ok=True)

        submission = Submission(test_id=test_id, student_id=current_user.id, status="pending")
        db.session.add(submission)
        db.session.flush()

        for page_num, file in enumerate(files, start=1):
            filename = f"submission_{submission.id}_page{page_num}{os.path.splitext(file.filename)[1]}"
            filepath = os.path.join(upload_dir, filename)
            file.save(filepath)
            db.session.add(SubmissionPage(submission_id=submission.id, page_number=page_num, image_path=filepath))

        db.session.commit()
        log(current_user.id, f"Submitted answer for test {test_id} ({len(files)} page(s))")

        # Run pipeline in background thread — never block the HTTP response
        import threading
        t = threading.Thread(target=_run_pipeline, args=(submission.id,))
        t.daemon = True
        t.start()

        return jsonify({"message": "Submitted successfully", "submission_id": submission.id, "status": "pending", "pages": len(files)}), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "Failed to save submission"}), 500

@student.route("/submission/status/<int:submission_id>", methods=["GET"])
@login_required
def submission_status(submission_id):
    submission = Submission.query.get_or_404(submission_id)
    if current_user.role == "student" and submission.student_id != current_user.id:
        return jsonify({"error": "Unauthorized"}), 403
    return jsonify({
        "submission_id": submission.id, "status": submission.status,
        "uploaded_at": submission.uploaded_at.isoformat(), "test_id": submission.test_id
    }), 200

@student.route("/results/<int:submission_id>", methods=["GET"])
@login_required
def get_results(submission_id):
    submission = Submission.query.get_or_404(submission_id)
    if current_user.role == "student" and submission.student_id != current_user.id:
        return jsonify({"error": "Unauthorized"}), 403
    if submission.status != "published":
        return jsonify({"error": "Results not yet published", "status": submission.status}), 403

    evaluations = Evaluation.query.filter_by(submission_id=submission_id).all()
    results, total_scored, total_max = [], 0, 0
    for ev in evaluations:
        q = Question.query.get(ev.question_id)
        results.append({
            "question_number": q.question_number, "question_text": q.question_text,
            "max_marks": q.max_marks, "final_score": ev.final_score,
            "ai_score": ev.ai_score, "override_score": ev.teacher_score,
            "ocr_text": ev.extracted_answer, "feedback": ev.ai_feedback
        })
        total_scored += ev.final_score or 0
        total_max    += q.max_marks

    test = Test.query.get(submission.test_id)
    log(current_user.id, f"Viewed results for submission {submission_id}")
    return jsonify({
        "submission_id": submission_id, "test_id": submission.test_id,
        "test_title": test.title if test else "",
        "status": submission.status, "total_score": total_scored,
        "max_total": total_max,
        "percentage": round((total_scored / total_max) * 100, 2) if total_max else 0,
        "questions": results
    }), 200

@student.route("/student/performance", methods=["GET"])
@login_required
def get_performance():
    if current_user.role != "student":
        return jsonify({"error": "Students only"}), 403
    subs = Submission.query.filter_by(student_id=current_user.id).order_by(Submission.uploaded_at).all()
    results = []
    for sub in subs:
        test  = Test.query.get(sub.test_id)
        evals = Evaluation.query.filter_by(submission_id=sub.id).all()
        total = sum(e.final_score or 0 for e in evals)
        max_m = test.total_marks if test else 0
        results.append({
            "test_id": sub.test_id, "test_title": test.title if test else "Unknown",
            "subject": test.subject if test else "",
            "score": total, "max_marks": max_m, "status": sub.status,
            "percentage": round((total / max_m) * 100, 1) if max_m and sub.status == "published" else None,
            "submitted_at": sub.uploaded_at.isoformat()
        })
    published = [r for r in results if r["percentage"] is not None]
    avg = round(sum(r["percentage"] for r in published) / len(published), 1) if published else 0
    return jsonify({"results": results, "average_percentage": avg, "total_tests": len(results)}), 200

# ── NOTIFICATIONS ───────────────────────────────────────────
@student.route("/student/notifications", methods=["GET"])
@login_required
def get_notifications():
    notifs = Notification.query.filter_by(user_id=current_user.id)\
               .order_by(Notification.created_at.desc()).limit(20).all()
    return jsonify({"notifications": [
        {"id": n.id, "message": n.message, "is_read": n.is_read,
         "created_at": n.created_at.isoformat()} for n in notifs
    ], "unread_count": Notification.query.filter_by(user_id=current_user.id, is_read=False).count()}), 200

@student.route("/student/notifications/read", methods=["POST"])
@login_required
def mark_all_read():
    Notification.query.filter_by(user_id=current_user.id, is_read=False).update({"is_read": True})
    db.session.commit()
    return jsonify({"message": "All marked as read"}), 200

# ── ATTEMPT HISTORY ─────────────────────────────────────────
@student.route("/student/history", methods=["GET"])
@login_required
def get_history():
    if current_user.role != "student":
        return jsonify({"error": "Students only"}), 403
    subs = Submission.query.filter_by(student_id=current_user.id)\
             .order_by(Submission.uploaded_at.desc()).all()
    history = []
    for sub in subs:
        test  = Test.query.get(sub.test_id)
        evals = Evaluation.query.filter_by(submission_id=sub.id).all()
        total = sum(e.final_score or 0 for e in evals)
        max_m = test.total_marks if test else 0
        history.append({
            "submission_id": sub.id,
            "test_id": sub.test_id,
            "test_title": test.title if test else "Unknown",
            "subject": test.subject if test else "",
            "score": total if sub.status == "published" else None,
            "max_marks": max_m,
            "percentage": round((total/max_m)*100,1) if max_m and sub.status=="published" else None,
            "status": sub.status,
            "submitted_at": sub.uploaded_at.isoformat()
        })
    return jsonify({"history": history}), 200

# ── CLASS AVERAGE ────────────────────────────────────────────
@student.route("/student/class-average/<int:test_id>", methods=["GET"])
@login_required
def class_average(test_id):
    test = Test.query.get_or_404(test_id)
    subs = Submission.query.filter_by(test_id=test_id, status="published").all()
    if not subs:
        return jsonify({"class_average": 0, "total_students": 0, "highest": 0, "lowest": 0}), 200
    scores = []
    for sub in subs:
        evals = Evaluation.query.filter_by(submission_id=sub.id).all()
        total = sum(e.final_score or 0 for e in evals)
        scores.append(round((total / test.total_marks) * 100, 1) if test.total_marks else 0)
    return jsonify({
        "class_average": round(sum(scores)/len(scores), 1),
        "total_students": len(scores),
        "highest": max(scores),
        "lowest": min(scores)
    }), 200

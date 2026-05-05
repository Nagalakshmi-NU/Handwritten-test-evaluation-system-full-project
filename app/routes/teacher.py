from flask import Blueprint, request, jsonify, send_file
from flask_login import login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from app import db
from app.models import Test, Question, Submission, Evaluation, User, ActivityLog, Notification
from app.utils.pdf_generator import generate_answer_sheet
import json, os, io, csv
from datetime import datetime

teacher = Blueprint("teacher", __name__)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def log(user_id, action):
    db.session.add(ActivityLog(user_id=user_id, action=action))
    db.session.commit()

def teacher_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if current_user.role not in ("teacher", "admin"):
            return jsonify({"error": "Unauthorized"}), 403
        return f(*args, **kwargs)
    return decorated

@teacher.route("/test/create", methods=["POST"])
@login_required
@teacher_required
def create_test():
    data = request.get_json()
    if not data.get("title") or not data.get("questions"):
        return jsonify({"error": "title and questions are required"}), 400

    total_marks = sum(q["max_marks"] for q in data["questions"])
    test = Test(
        teacher_id=current_user.id,
        title=data["title"],
        subject=data.get("subject", ""),
        total_marks=total_marks,
        status="active"
    )
    db.session.add(test)
    db.session.flush()

    questions_for_pdf = []
    for q in data["questions"]:
        question = Question(
            test_id=test.id,
            question_number=q["question_number"],
            question_text=q.get("question_text", ""),
            model_answer=q["model_answer"],
            max_marks=q["max_marks"],
            rubric_json=json.dumps(q["rubric"])
        )
        db.session.add(question)
        questions_for_pdf.append({
            "question_number": q["question_number"],
            "question_text": q.get("question_text", f"Question {q['question_number']}"),
            "max_marks": q["max_marks"],
            "box_height": q.get("box_height", 120)
        })

    db.session.flush()
    pdf_path = os.path.join(BASE_DIR, 'generated_pdfs', f'test_{test.id}_answersheet.pdf')
    os.makedirs(os.path.join(BASE_DIR, 'generated_pdfs'), exist_ok=True)

    template = data.get("template", {})
    bounding_boxes = generate_answer_sheet(
        test_id=test.id, title=data["title"],
        questions=questions_for_pdf, output_path=pdf_path,
        page_size=template.get("page_size", "A4"),
        orientation=template.get("orientation", "portrait"),
        margin=int(template.get("margin", 40)),
        aruco_size=int(template.get("aruco_size", 30)),
        aruco_dict=template.get("aruco_dict", "DICT_4X4_50"),
        aruco_start_id=int(template.get("aruco_start_id", 0))
    )

    for bbox in bounding_boxes:
        q_record = Question.query.filter_by(test_id=test.id, question_number=bbox["question_number"]).first()
        if q_record:
            q_record.rubric_json = json.dumps({"rubric": json.loads(q_record.rubric_json), "bbox": bbox})

    db.session.commit()
    log(current_user.id, f"Created test '{data['title']}' (ID:{test.id})")
    return jsonify({
        "message": "Test created successfully", "test_id": test.id,
        "total_marks": total_marks, "pdf_url": f"/test/{test.id}/download-sheet",
        "bounding_boxes": bounding_boxes
    }), 201

@teacher.route("/test/<int:test_id>/upload-paper", methods=["POST"])
@login_required
@teacher_required
def upload_paper(test_id):
    test = Test.query.get_or_404(test_id)
    if test.teacher_id != current_user.id:
        return jsonify({"error": "Unauthorized"}), 403
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    paper_type = request.form.get("type", "question")  # 'question' or 'answer'
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".pdf"]:
        return jsonify({"error": "Invalid file type"}), 400
    upload_dir = os.path.join(BASE_DIR, "uploads", "papers")
    os.makedirs(upload_dir, exist_ok=True)
    filename = f"test_{test_id}_{paper_type}_paper{ext}"
    filepath = os.path.join(upload_dir, filename)
    file.save(filepath)

    # If answer paper uploaded for Method 2, extract text and update model answers
    if paper_type == "answer":
        try:
            import sys
            search_path = BASE_DIR
            if os.path.exists(os.path.join(search_path, 'ml_pipeline.py')):
                sys.path.insert(0, search_path)

            from ml_pipeline import _load_trocr
            import cv2, numpy as np
            from PIL import Image as PILImage

            if ext == ".pdf":
                import fitz
                doc = fitz.open(filepath)
                pix = doc[0].get_pixmap(dpi=150)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if pix.n == 3 else cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            else:
                img = cv2.imread(filepath)

            if img is not None:
                questions = Question.query.filter_by(test_id=test_id).order_by(Question.question_number).all()
                if questions:
                    processor, model = _load_trocr()
                    h = img.shape[0]
                    section_h = h // len(questions)
                    for i, q in enumerate(questions):
                        crop = img[i * section_h:(i + 1) * section_h, :]
                        try:
                            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            pil = PILImage.fromarray(rgb)
                            pixel_values = processor(images=pil, return_tensors="pt").pixel_values
                            generated = model.generate(pixel_values, max_new_tokens=256)
                            extracted = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
                            if extracted:
                                rubric_data = json.loads(q.rubric_json)
                                rubric = rubric_data.get("rubric", rubric_data) if isinstance(rubric_data, dict) else rubric_data
                                q.model_answer = extracted
                                q.rubric_json = json.dumps({
                                    "rubric": rubric if isinstance(rubric, list) else [str(rubric)],
                                    "bbox": rubric_data.get("bbox", {}) if isinstance(rubric_data, dict) else {}
                                })
                        except Exception as e:
                            print(f"OCR failed for Q{q.question_number}: {e}")
                    db.session.commit()
        except Exception as e:
            print(f"Answer paper OCR failed: {e}")

    # If question paper uploaded, extract question texts using OCR
    if paper_type == "question":
        try:
            import sys
            search_path = BASE_DIR
            if os.path.exists(os.path.join(search_path, 'ml_pipeline.py')):
                sys.path.insert(0, search_path)

            from ml_pipeline import _load_trocr
            import cv2, numpy as np
            from PIL import Image as PILImage

            if ext == ".pdf":
                import fitz
                doc = fitz.open(filepath)
                pix = doc[0].get_pixmap(dpi=150)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if pix.n == 3 else cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            else:
                img = cv2.imread(filepath)

            if img is not None:
                questions = Question.query.filter_by(test_id=test_id).order_by(Question.question_number).all()
                if questions:
                    processor, model = _load_trocr()
                    h = img.shape[0]
                    section_h = h // len(questions)
                    for i, q in enumerate(questions):
                        crop = img[i * section_h:(i + 1) * section_h, :]
                        try:
                            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            pil = PILImage.fromarray(rgb)
                            pixel_values = processor(images=pil, return_tensors="pt").pixel_values
                            generated = model.generate(pixel_values, max_new_tokens=256)
                            extracted = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
                            if extracted:
                                q.question_text = extracted
                        except Exception as e:
                            print(f"Question OCR failed for Q{q.question_number}: {e}")
                    db.session.commit()
        except Exception as e:
            print(f"Question paper OCR failed: {e}")

    log(current_user.id, f"Uploaded {paper_type} paper for test {test_id}")
    return jsonify({"message": f"{paper_type.capitalize()} paper uploaded", "path": filepath}), 200


@teacher.route("/test/<int:test_id>/view-paper/<paper_type>", methods=["GET"])
@login_required
def view_paper(test_id, paper_type):
    """Serve uploaded question or answer paper file."""
    upload_dir = os.path.join(BASE_DIR, "uploads", "papers")
    for ext in [".pdf", ".jpg", ".jpeg", ".png"]:
        filepath = os.path.join(upload_dir, f"test_{test_id}_{paper_type}_paper{ext}")
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=False)
    return jsonify({"error": "File not found"}), 404


@teacher.route("/test/<int:test_id>/download-sheet", methods=["GET"])
@login_required
def download_sheet(test_id):
    pdf_path = os.path.join(BASE_DIR, 'generated_pdfs', f'test_{test_id}_answersheet.pdf')
    if not os.path.exists(pdf_path):
        return jsonify({"error": "PDF not found"}), 404
    return send_file(pdf_path, as_attachment=True, download_name=f"answersheet_test_{test_id}.pdf")

@teacher.route("/teacher/tests", methods=["GET"])
@login_required
@teacher_required
def get_teacher_tests():
    tests = Test.query.filter_by(teacher_id=current_user.id).order_by(Test.created_at.desc()).all()
    return jsonify({"tests": [{
        "test_id": t.id, "title": t.title, "subject": t.subject,
        "total_marks": t.total_marks, "status": t.status,
        "submissions": Submission.query.filter_by(test_id=t.id).count(),
        "created_at": t.created_at.isoformat()
    } for t in tests]}), 200

@teacher.route("/queue", methods=["GET"])
@login_required
@teacher_required
def get_queue():
    subs = Submission.query.join(Test, Submission.test_id == Test.id)\
        .filter(Test.teacher_id == current_user.id)\
        .order_by(Submission.uploaded_at.desc()).all()
    result = []
    for sub in subs:
        student = User.query.get(sub.student_id)
        test    = Test.query.get(sub.test_id)
        result.append({
            "submission_id": sub.id, "test_id": sub.test_id,
            "test_title": test.title if test else "",
            "student_name": student.name if student else "Unknown",
            "status": sub.status, "uploaded_at": sub.uploaded_at.isoformat()
        })
    return jsonify({"queue": result}), 200

@teacher.route("/review/<int:submission_id>", methods=["GET"])
@login_required
@teacher_required
def get_review(submission_id):
    submission = Submission.query.get_or_404(submission_id)
    test = Test.query.get(submission.test_id)
    if test.teacher_id != current_user.id and current_user.role != "admin":
        return jsonify({"error": "Unauthorized"}), 403

    student = User.query.get(submission.student_id)
    evaluations = Evaluation.query.filter_by(submission_id=submission_id).all()
    result = []
    for ev in evaluations:
        question = Question.query.get(ev.question_id)
        rubric_data = json.loads(question.rubric_json)
        rubric = rubric_data.get("rubric", rubric_data) if isinstance(rubric_data, dict) else rubric_data
        bbox   = rubric_data.get("bbox", {}) if isinstance(rubric_data, dict) else {}
        result.append({
            "question_number": question.question_number,
            "question_text": question.question_text,
            "max_marks": question.max_marks, "rubric": rubric,
            "ocr_text": ev.extracted_answer,
            "ai_score": ev.ai_score or 0,
            "feedback": ev.ai_feedback,
            "confidence": ev.confidence or 0,
            "override_score": ev.teacher_score,
            "final_score": ev.final_score or 0,
            "bbox": bbox
        })

    log(current_user.id, f"Reviewed submission {submission_id}")
    return jsonify({
        "submission_id": submission_id, "status": submission.status,
        "test_id": submission.test_id, "test_title": test.title,
        "student_name": student.name if student else "Unknown",
        "uploaded_at": submission.uploaded_at.isoformat(),
        "questions": result
    }), 200

@teacher.route("/review/save", methods=["POST"])
@login_required
@teacher_required
def save_review():
    data = request.get_json()
    submission_id   = data.get("submission_id")
    question_number = data.get("question_number")
    override_score  = data.get("override_score")

    if submission_id is None or question_number is None or override_score is None:
        return jsonify({"error": "submission_id, question_number and override_score required"}), 400

    submission = Submission.query.get_or_404(submission_id)
    test = Test.query.get(submission.test_id)
    if test.teacher_id != current_user.id and current_user.role != "admin":
        return jsonify({"error": "Unauthorized"}), 403

    question = Question.query.filter_by(test_id=submission.test_id, question_number=question_number).first()
    if not question:
        return jsonify({"error": "Question not found"}), 404

    evaluation = Evaluation.query.filter_by(submission_id=submission_id, question_id=question.id).first()
    if not evaluation:
        return jsonify({"error": "Evaluation not found"}), 404

    score = max(0, min(float(override_score), question.max_marks))
    evaluation.teacher_score = score
    evaluation.final_score   = score
    db.session.commit()
    log(current_user.id, f"Overrode score for Q{question_number} in submission {submission_id} → {score}")
    return jsonify({"message": "Score saved", "final_score": score}), 200

@teacher.route("/publish/<int:submission_id>", methods=["POST"])
@login_required
@teacher_required
def publish(submission_id):
    submission = Submission.query.get_or_404(submission_id)
    test = Test.query.get(submission.test_id)
    if test.teacher_id != current_user.id and current_user.role != "admin":
        return jsonify({"error": "Unauthorized"}), 403
    if submission.status not in ["llm_done", "reviewed"]:
        return jsonify({"error": f"Cannot publish. Status: {submission.status}"}), 400

    evaluations = Evaluation.query.filter_by(submission_id=submission_id).all()
    if not evaluations:
        return jsonify({"error": "No evaluations found"}), 400

    total = 0
    for ev in evaluations:
        ev.final_score = ev.teacher_score if ev.teacher_score is not None else (ev.ai_score or 0)
        total += ev.final_score

    submission.status = "published"
    db.session.commit()
    # send notification to student
    from app.models import Notification
    test = Test.query.get(submission.test_id)
    db.session.add(Notification(
        user_id=submission.student_id,
        message=f"🎉 Your results for '{test.title if test else 'Test'}' have been published! Check your scores now."
    ))
    db.session.commit()
    log(current_user.id, f"Published results for submission {submission_id}")
    return jsonify({"message": "Published", "submission_id": submission_id, "total_final_score": total}), 200

@teacher.route("/teacher/students", methods=["GET"])
@login_required
@teacher_required
def get_teacher_students():
    """All students who submitted to this teacher's tests"""
    tests = Test.query.filter_by(teacher_id=current_user.id).all()
    test_ids = [t.id for t in tests]
    if not test_ids:
        return jsonify({"students": [], "top3": []}), 200

    subs = Submission.query.filter(Submission.test_id.in_(test_ids)).all()
    student_map = {}
    for sub in subs:
        sid = sub.student_id
        if sid not in student_map:
            student = User.query.get(sid)
            student_map[sid] = {
                "student_id": sid,
                "name": student.name if student else "Unknown",
                "email": student.email if student else "",
                "student_id_no": student.student_id if student else "",
                "total_submissions": 0, "published": 0, "avg_percentage": 0,
                "scores": []
            }
        student_map[sid]["total_submissions"] += 1
        if sub.status == "published":
            student_map[sid]["published"] += 1
            evals = Evaluation.query.filter_by(submission_id=sub.id).all()
            test  = Test.query.get(sub.test_id)
            total = sum(e.final_score or 0 for e in evals)
            pct   = round((total / test.total_marks) * 100, 1) if test and test.total_marks else 0
            student_map[sid]["scores"].append(pct)

    students = []
    for s in student_map.values():
        s["avg_percentage"] = round(sum(s["scores"]) / len(s["scores"]), 1) if s["scores"] else 0
        del s["scores"]
        students.append(s)

    students.sort(key=lambda x: x["avg_percentage"], reverse=True)
    top3 = students[:3]
    return jsonify({"students": students, "top3": top3}), 200

@teacher.route("/teacher/student/<int:student_id>/profile", methods=["GET"])
@login_required
@teacher_required
def get_student_profile(student_id):
    """Full profile of a student for this teacher"""
    tests = Test.query.filter_by(teacher_id=current_user.id).all()
    test_ids = [t.id for t in tests]
    student = User.query.get_or_404(student_id)

    subs = Submission.query.filter(
        Submission.student_id == student_id,
        Submission.test_id.in_(test_ids)
    ).order_by(Submission.uploaded_at).all()

    results = []
    for sub in subs:
        test  = Test.query.get(sub.test_id)
        evals = Evaluation.query.filter_by(submission_id=sub.id).all()
        total = sum(e.final_score or 0 for e in evals)
        max_m = test.total_marks if test else 0
        pct   = round((total / max_m) * 100, 1) if max_m and sub.status == "published" else None
        results.append({
            "submission_id": sub.id, "test_id": sub.test_id,
            "test_title": test.title if test else "Unknown",
            "subject": test.subject if test else "",
            "score": total, "max_marks": max_m,
            "percentage": pct, "status": sub.status,
            "submitted_at": sub.uploaded_at.isoformat()
        })

    published = [r for r in results if r["percentage"] is not None]
    avg = round(sum(r["percentage"] for r in published) / len(published), 1) if published else 0
    best = max((r["percentage"] for r in published), default=0)

    # existing feedback
    from app.models import TeacherFeedback
    feedbacks = TeacherFeedback.query.filter_by(
        teacher_id=current_user.id, student_id=student_id
    ).order_by(TeacherFeedback.created_at.desc()).all()

    return jsonify({
        "student": {
            "id": student.id, "name": student.name, "email": student.email,
            "student_id_no": student.student_id
        },
        "results": results,
        "avg_percentage": avg, "best_score": best,
        "total_tests": len(results),
        "feedbacks": [{"id": f.id, "message": f.message,
                       "created_at": f.created_at.isoformat()} for f in feedbacks]
    }), 200

@teacher.route("/teacher/feedback", methods=["POST"])
@login_required
@teacher_required
def give_feedback():
    data = request.get_json()
    student_id = data.get("student_id")
    message    = data.get("message", "").strip()
    if not student_id or not message:
        return jsonify({"error": "student_id and message required"}), 400
    from app.models import TeacherFeedback
    fb = TeacherFeedback(teacher_id=current_user.id, student_id=student_id, message=message)
    db.session.add(fb)
    db.session.commit()
    log(current_user.id, f"Gave feedback to student {student_id}")
    return jsonify({"message": "Feedback sent", "id": fb.id}), 201

@teacher.route("/teacher/feedback/<int:feedback_id>", methods=["DELETE"])
@login_required
@teacher_required
def delete_feedback(feedback_id):
    from app.models import TeacherFeedback
    fb = TeacherFeedback.query.get_or_404(feedback_id)
    if fb.teacher_id != current_user.id:
        return jsonify({"error": "Unauthorized"}), 403
    db.session.delete(fb)
    db.session.commit()
    return jsonify({"message": "Deleted"}), 200

@teacher.route("/teacher/reports", methods=["GET"])
@login_required
@teacher_required
def get_reports():
    tests = Test.query.filter_by(teacher_id=current_user.id).all()
    report = []
    for t in tests:
        subs = Submission.query.filter_by(test_id=t.id, status="published").all()
        scores = []
        for sub in subs:
            evals = Evaluation.query.filter_by(submission_id=sub.id).all()
            total = sum(e.final_score or 0 for e in evals)
            scores.append(round((total / t.total_marks) * 100, 1) if t.total_marks else 0)
        report.append({
            "test_id": t.id, "title": t.title, "subject": t.subject,
            "total_submissions": len(subs),
            "class_average": round(sum(scores) / len(scores), 1) if scores else 0,
            "highest": max(scores) if scores else 0,
            "lowest": min(scores) if scores else 0
        })
    return jsonify({"reports": report}), 200

@teacher.route("/test/<int:test_id>/close", methods=["POST"])
@login_required
@teacher_required
def close_test(test_id):
    test = Test.query.get_or_404(test_id)
    if test.teacher_id != current_user.id:
        return jsonify({"error": "Unauthorized"}), 403
    test.status = "closed" if test.status == "active" else "active"
    db.session.commit()
    log(current_user.id, f"Changed test '{test.title}' to {test.status}")
    return jsonify({"message": f"Test {test.status}", "status": test.status}), 200

@teacher.route("/test/<int:test_id>/deadline", methods=["POST"])
@login_required
@teacher_required
def set_deadline(test_id):
    test = Test.query.get_or_404(test_id)
    if test.teacher_id != current_user.id:
        return jsonify({"error": "Unauthorized"}), 403
    data = request.get_json()
    deadline = data.get("deadline")
    test.deadline = datetime.fromisoformat(deadline) if deadline else None
    db.session.commit()
    log(current_user.id, f"Set deadline for test '{test.title}'")
    return jsonify({"message": "Deadline updated"}), 200

@teacher.route("/submission/<int:submission_id>/reevaluate", methods=["POST"])
@login_required
@teacher_required
def reevaluate(submission_id):
    submission = Submission.query.get_or_404(submission_id)
    test = Test.query.get(submission.test_id)
    if test.teacher_id != current_user.id:
        return jsonify({"error": "Unauthorized"}), 403
    # Reset evaluations so pipeline re-runs fresh
    from app.models import Evaluation
    Evaluation.query.filter_by(submission_id=submission_id).delete()
    submission.status = "pending"
    db.session.commit()
    # Re-run pipeline
    from app.tasks.ocr_tasks import _run_pipeline
    import threading
    t = threading.Thread(target=_run_pipeline, args=(submission_id,))
    t.daemon = True
    t.start()
    log(current_user.id, f"Re-evaluation triggered for submission {submission_id}")
    return jsonify({"message": "Re-evaluation started"}), 200

@teacher.route("/teacher/export/<int:test_id>", methods=["GET"])
@login_required
@teacher_required
def export_results(test_id):
    from flask import Response
    test = Test.query.get_or_404(test_id)
    if test.teacher_id != current_user.id:
        return jsonify({"error": "Unauthorized"}), 403

    # Get all approved students
    all_students = User.query.filter_by(role="student", status="approved").all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Student Name", "Student ID", "Email",
                     "Total Score", "Max Marks", "Percentage", "Grade", "Status"])

    for student in all_students:
        sub = Submission.query.filter_by(test_id=test_id, student_id=student.id).first()
        if sub:
            evals = Evaluation.query.filter_by(submission_id=sub.id).all()
            total = sum(e.final_score or 0 for e in evals)
            if sub.status == "published" and test.total_marks:
                pct   = round((total / test.total_marks) * 100, 1)
                grade = "A" if pct >= 80 else "B" if pct >= 60 else "C" if pct >= 40 else "D"
            else:
                pct   = "—"
                grade = "—"
                total = "—"
            status = sub.status.replace("_", " ").title()
        else:
            total  = "—"
            pct    = "—"
            grade  = "—"
            status = "Not Submitted"

        writer.writerow([student.name, student.student_id or "—",
                         student.email, total, test.total_marks, pct, grade, status])

    return Response(output.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment;filename=results_test_{test_id}.csv"})

@teacher.route("/teacher/announce", methods=["POST"])
@login_required
@teacher_required
def announce():
    data    = request.get_json()
    message = data.get("message", "").strip()
    test_id = data.get("test_id")
    if not message:
        return jsonify({"error": "Message required"}), 400
    from app.models import Announcement
    ann = Announcement(teacher_id=current_user.id, test_id=test_id, message=message)
    db.session.add(ann)
    if test_id:
        subs = Submission.query.filter_by(test_id=test_id).all()
        student_ids = list({s.student_id for s in subs})
    else:
        tests = Test.query.filter_by(teacher_id=current_user.id).all()
        test_ids = [t.id for t in tests]
        subs = Submission.query.filter(Submission.test_id.in_(test_ids)).all()
        student_ids = list({s.student_id for s in subs})
    for sid in student_ids:
        db.session.add(Notification(user_id=sid, message=f"📢 {current_user.name}: {message}"))
    db.session.commit()
    log(current_user.id, f"Sent announcement to {len(student_ids)} students")
    return jsonify({"message": f"Sent to {len(student_ids)} students"}), 200

# ── Question Bank ────────────────────────────────────────────
@teacher.route("/teacher/question-bank", methods=["GET"])
@login_required
@teacher_required
def get_question_bank():
    from app.models import QuestionBank
    import json
    subject = request.args.get("subject")
    q = QuestionBank.query.filter_by(teacher_id=current_user.id)
    if subject: q = q.filter_by(subject=subject)
    items = q.order_by(QuestionBank.created_at.desc()).all()
    return jsonify({"questions": [{
        "id": i.id, "subject": i.subject, "question_text": i.question_text,
        "model_answer": i.model_answer, "max_marks": i.max_marks,
        "rubric": json.loads(i.rubric_json), "created_at": i.created_at.isoformat()
    } for i in items]}), 200

@teacher.route("/teacher/question-bank", methods=["POST"])
@login_required
@teacher_required
def add_to_question_bank():
    from app.models import QuestionBank
    import json
    data = request.get_json()
    if not data.get("question_text") or not data.get("model_answer"):
        return jsonify({"error": "question_text and model_answer required"}), 400
    item = QuestionBank(
        teacher_id=current_user.id, subject=data.get("subject", ""),
        question_text=data["question_text"], model_answer=data["model_answer"],
        max_marks=data.get("max_marks", 5),
        rubric_json=json.dumps(data.get("rubric", []))
    )
    db.session.add(item)
    db.session.commit()
    log(current_user.id, f"Added question to bank: {data['question_text'][:40]}")
    return jsonify({"message": "Added to question bank", "id": item.id}), 201

@teacher.route("/teacher/question-bank/<int:qid>", methods=["DELETE"])
@login_required
@teacher_required
def delete_from_question_bank(qid):
    from app.models import QuestionBank
    item = QuestionBank.query.get_or_404(qid)
    if item.teacher_id != current_user.id:
        return jsonify({"error": "Unauthorized"}), 403
    db.session.delete(item)
    db.session.commit()
    return jsonify({"message": "Deleted"}), 200

@teacher.route("/student/announcements", methods=["GET"])
@login_required
def get_announcements():
    from app.models import Announcement
    subs = Submission.query.filter_by(student_id=current_user.id).all()
    test_ids = list({s.test_id for s in subs})
    anns = Announcement.query.filter(Announcement.test_id.in_(test_ids))\
             .order_by(Announcement.created_at.desc()).limit(20).all()
    return jsonify({"announcements": [{
        "id": a.id, "message": a.message,
        "teacher": User.query.get(a.teacher_id).name if User.query.get(a.teacher_id) else "Teacher",
        "created_at": a.created_at.isoformat()
    } for a in anns]}), 200

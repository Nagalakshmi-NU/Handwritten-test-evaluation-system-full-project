"""
app/tasks/ocr_tasks.py
Background processing pipeline.
Uses ML PART/pipeline.py when available, falls back to local evaluation.
"""

import sys, os, json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

ML_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..'
))
if os.path.exists(os.path.join(ML_PATH, 'ml_pipeline.py')):
    sys.path.insert(0, ML_PATH)
    ML_AVAILABLE = True
else:
    ML_AVAILABLE = False


def _run_pipeline(submission_id):
    """Core pipeline — runs directly without Celery."""
    from app import create_app, db
    from app.models import Submission, Question, Evaluation
    from config import Config

    app = create_app()
    with app.app_context():
        submission = db.session.get(Submission, submission_id)
        if not submission:
            return {"error": "Submission not found"}

        submission.status = "processing"
        db.session.commit()

        questions = Question.query.filter_by(test_id=submission.test_id).all()
        if not questions:
            submission.status = "failed"
            db.session.commit()
            return {"error": "No questions found"}

        # Get image path from submission pages
        from app.models import SubmissionPage
        page = SubmissionPage.query.filter_by(
            submission_id=submission_id, page_number=1
        ).first()
        image_path = page.image_path if page else None

        # Build questions list for ML pipeline
        questions_data = []
        bounding_boxes = []
        for q in questions:
            rubric_data = json.loads(q.rubric_json)
            rubric = rubric_data.get("rubric", rubric_data) if isinstance(rubric_data, dict) else rubric_data
            bbox   = rubric_data.get("bbox", None) if isinstance(rubric_data, dict) else None

            questions_data.append({
                "question_number": q.question_number,
                "question_text":   q.question_text or f"Question {q.question_number}",
                "model_answer":    q.model_answer,
                "max_marks":       q.max_marks,
                "rubric":          rubric,
                "question_id":     q.id
            })
            if bbox:
                bounding_boxes.append(bbox)

        # ── Try ML PART pipeline ─────────────────────────────
        ml_results = None
        if ML_AVAILABLE and image_path and os.path.exists(image_path):
            try:
                from ml_pipeline import process_submission as ml_process
                ml_output = ml_process(
                    submission_id=submission_id,
                    image_path=image_path,
                    questions=questions_data,
                    bounding_boxes=bounding_boxes,
                    groq_api_key=Config.GROQ_API_KEY
                )
                if ml_output["status"] == "success":
                    ml_results = {r["question_number"]: r for r in ml_output["results"]}
                    print(f"ML pipeline succeeded for submission {submission_id}")
            except Exception as e:
                print(f"ML pipeline failed: {e}, falling back to local evaluation")

        # ── Fallback: local evaluation ───────────────────────
        if not ml_results:
            from app.utils.llm_evaluator import evaluate_answer

            ml_results = {}
            for q in questions_data:
                q_num = q["question_number"]

                # Try real OCR on the uploaded image
                student_answer = "No answer provided"
                if image_path and os.path.exists(image_path):
                    try:
                        import cv2
                        import numpy as np
                        from PIL import Image

                        img = cv2.imread(image_path)
                        if img is None and image_path.lower().endswith('.pdf'):
                            import fitz
                            doc = fitz.open(image_path)
                            pix = doc[0].get_pixmap(dpi=150)
                            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                            img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if pix.n == 3 else cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

                        if img is not None:
                            # Use ml_pipeline lazy loader to avoid reloading model per question
                            from ml_pipeline import _load_trocr
                            processor, model = _load_trocr()
                            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            pil = Image.fromarray(rgb)
                            pixel_values = processor(images=pil, return_tensors="pt").pixel_values
                            generated = model.generate(pixel_values, max_new_tokens=256)
                            student_answer = processor.batch_decode(generated, skip_special_tokens=True)[0].strip() or "No answer detected"
                    except Exception as e:
                        print(f"Fallback OCR failed for Q{q_num}: {e}")

                try:
                    result = evaluate_answer(
                        question_text=q["question_text"],
                        model_answer=q["model_answer"],
                        rubric=q["rubric"],
                        student_answer=student_answer,
                        max_marks=q["max_marks"],
                        groq_api_key=Config.GROQ_API_KEY
                    )
                except Exception as e:
                    result = {
                        "score": 0,
                        "confidence": 0.5,
                        "feedback": "Evaluation failed. Teacher review recommended.",
                        "matched_points": [], "missing_points": []
                    }
                ml_results[q_num] = {
                    "question_number": q_num,
                    "extracted_text":  student_answer,
                    "score":           result["score"],
                    "feedback":        result["feedback"],
                    "confidence":      result["confidence"]
                }

        # ── Save evaluations to DB ───────────────────────────
        for q in questions_data:
            q_num  = q["question_number"]
            res    = ml_results.get(q_num, {})
            ev = Evaluation.query.filter_by(
                submission_id=submission_id,
                question_id=q["question_id"]
            ).first()

            if ev:
                ev.extracted_answer = res.get("extracted_text", "")
                ev.ai_score         = res.get("score", 0)
                ev.ai_feedback      = res.get("feedback", "")
                ev.confidence       = res.get("confidence", 0.5)
                ev.final_score      = res.get("score", 0)
            else:
                db.session.add(Evaluation(
                    submission_id=submission_id,
                    question_id=q["question_id"],
                    extracted_answer=res.get("extracted_text", ""),
                    ai_score=res.get("score", 0),
                    ai_feedback=res.get("feedback", ""),
                    confidence=res.get("confidence", 0.5),
                    final_score=res.get("score", 0)
                ))

        submission.status = "llm_done"
        db.session.commit()
        print(f"Submission {submission_id} processed successfully")
        return {"submission_id": submission_id, "status": "llm_done"}


# Try Celery, fall back to direct execution
try:
    from celery import Celery

    def make_celery():
        return Celery("tasks",
                      broker="redis://localhost:6379/0",
                      backend="redis://localhost:6379/0")

    celery = make_celery()

    @celery.task(bind=True, max_retries=3, default_retry_delay=10)
    def process_submission(self, submission_id):
        try:
            return _run_pipeline(submission_id)
        except Exception as e:
            try:
                raise self.retry(exc=e)
            except self.MaxRetriesExceededError:
                return {"error": str(e), "status": "failed"}

except Exception:
    def process_submission(submission_id):
        return _run_pipeline(submission_id)

    class _FakeTask:
        @staticmethod
        def delay(submission_id):
            import threading
            t = threading.Thread(target=_run_pipeline, args=(submission_id,))
            t.daemon = True
            t.start()

    process_submission.delay = _FakeTask.delay

"""
app/tasks/ocr_tasks.py
Render-proof pipeline: reads image from base64 stored in DB → Groq Vision OCR → Groq LLM evaluation
No dependency on disk files.
"""

import sys, os, json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

ML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if os.path.exists(os.path.join(ML_PATH, 'ml_pipeline.py')):
    sys.path.insert(0, ML_PATH)
    ML_AVAILABLE = True
else:
    ML_AVAILABLE = False


def _groq_vision_ocr(image_source, question_text, groq_api_key):
    """
    Run Groq Vision OCR on an image.
    image_source: file path OR base64 data URI string
    Returns extracted text string.
    """
    import base64
    from groq import Groq

    try:
        # Get base64 string
        if image_source.startswith('data:image'):
            # Already base64 data URI
            img_b64_uri = image_source
        elif os.path.exists(image_source):
            # Read from file
            with open(image_source, 'rb') as f:
                raw = f.read()
            ext = os.path.splitext(image_source)[1].lower().strip('.')
            if ext == 'jpg':
                ext = 'jpeg'
            img_b64 = base64.b64encode(raw).decode('utf-8')
            img_b64_uri = f"data:image/{ext};base64,{img_b64}"
        else:
            return "Image not available"

        client = Groq(api_key=groq_api_key)
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": img_b64_uri}
                    },
                    {
                        "type": "text",
                        "text": f"This is a student's handwritten answer sheet. Find and extract the answer written for this question: '{question_text}'. Return ONLY the handwritten answer text. If you cannot find an answer for this question, respond with 'No answer written'."
                    }
                ]
            }],
            max_tokens=512,
            temperature=0
        )
        text = response.choices[0].message.content.strip()
        return text if text else "No answer detected"
    except Exception as e:
        print(f"Groq Vision OCR error: {e}")
        return "OCR failed"


def _run_pipeline(submission_id):
    """Core pipeline — Render-proof, works with base64 from DB."""
    from app import create_app, db
    from app.models import Submission, Question, Evaluation, SubmissionPage
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

        page = SubmissionPage.query.filter_by(
            submission_id=submission_id, page_number=1
        ).first()
        image_path = page.image_path if page else None

        # Determine image source — prefer base64 from DB, fallback to file
        image_source = None
        if page and page.processed_text and page.processed_text.startswith('data:image'):
            image_source = page.processed_text  # base64 from DB
            print(f"Using base64 from DB for submission {submission_id}")
        elif image_path and os.path.exists(image_path):
            image_source = image_path  # file on disk
            print(f"Using file from disk for submission {submission_id}")
        else:
            print(f"No image available for submission {submission_id}")

        # Build questions data
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

        # Try ML pipeline first (works locally with file on disk)
        ml_results = None
        if ML_AVAILABLE and image_source and not image_source.startswith('data:image') and os.path.exists(image_source):
            try:
                from ml_pipeline import process_submission as ml_process
                ml_output = ml_process(
                    submission_id=submission_id,
                    image_path=image_source,
                    questions=questions_data,
                    bounding_boxes=bounding_boxes,
                    groq_api_key=Config.GROQ_API_KEY
                )
                if ml_output["status"] == "success":
                    ml_results = {r["question_number"]: r for r in ml_output["results"]}
                    print(f"ML pipeline succeeded for submission {submission_id}")
            except Exception as e:
                print(f"ML pipeline failed: {e}")

        # Groq Vision fallback (works on Render with base64 or file)
        if not ml_results:
            from app.utils.llm_evaluator import evaluate_answer
            ml_results = {}

            for q in questions_data:
                q_num = q["question_number"]

                # OCR
                if image_source and Config.GROQ_API_KEY:
                    student_answer = _groq_vision_ocr(image_source, q["question_text"], Config.GROQ_API_KEY)
                else:
                    student_answer = "No image available for OCR"

                # Evaluate
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
                        "score": 0, "confidence": 0.5,
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

        # Save to DB
        for q in questions_data:
            q_num = q["question_number"]
            res   = ml_results.get(q_num, {})
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


# Celery / threading fallback
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

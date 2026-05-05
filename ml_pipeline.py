"""
ml_pipeline.py  —  c:/h/
Full pipeline: ArUco alignment → answer-region crop → TrOCR → Groq LLM evaluation
Called by ocr_tasks.py via:  from ml_pipeline import process_submission
"""

import os, json, cv2, numpy as np
from PIL import Image

# ── TrOCR (lazy-loaded so startup is fast) ───────────────────
_trocr_processor = None
_trocr_model     = None

def _load_trocr():
    global _trocr_processor, _trocr_model
    if _trocr_processor is None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        print("Loading TrOCR model (first run may take a minute)...")
        _trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        _trocr_model     = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        print("TrOCR loaded.")
    return _trocr_processor, _trocr_model


# ── ArUco detection & perspective correction ─────────────────
def _detect_and_warp(image_bgr):
    """
    Detect 4 ArUco markers (IDs 0-3) and warp the page to a
    canonical A4-like rectangle (794 x 1123 px).
    Returns warped image or original if markers not found.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) < 4:
        return image_bgr, False

    id_to_corner = {}
    for i, mid in enumerate(ids.flatten()):
        if mid in (0, 1, 2, 3):
            c = corners[i][0]
            id_to_corner[mid] = c.mean(axis=0)  # centre of marker

    if len(id_to_corner) < 4:
        return image_bgr, False

    # marker layout: 0=TL, 1=TR, 2=BL, 3=BR
    src = np.float32([id_to_corner[0], id_to_corner[1],
                      id_to_corner[2], id_to_corner[3]])
    W, H = 794, 1123
    dst  = np.float32([[0, 0], [W, 0], [0, H], [W, H]])
    M    = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image_bgr, M, (W, H))
    return warped, True


# ── Crop answer region from bounding box ─────────────────────
def _crop_region(image_bgr, bbox, page_h):
    """
    bbox from pdf_generator: {x, y, width, height}
    y is measured from top of page (converted from PDF coords).
    """
    x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["width"]), int(bbox["height"])
    img_h, img_w = image_bgr.shape[:2]

    # Scale bbox to actual image dimensions
    scale_x = img_w / 794
    scale_y = img_h / 1123
    x  = int(x  * scale_x)
    y  = int(y  * scale_y)
    w  = int(w  * scale_x)
    h  = int(h  * scale_y)

    # Clamp
    x  = max(0, x)
    y  = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)

    crop = image_bgr[y:y2, x:x2]
    return crop if crop.size > 0 else image_bgr


# ── TrOCR inference on a single crop ─────────────────────────
def _ocr_crop(crop_bgr):
    processor, model = _load_trocr()
    rgb  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil  = Image.fromarray(rgb)

    # Resize if too small
    if pil.width < 32 or pil.height < 32:
        pil = pil.resize((max(pil.width, 128), max(pil.height, 64)))

    pixel_values = processor(images=pil, return_tensors="pt").pixel_values
    generated    = model.generate(pixel_values, max_new_tokens=256)
    text         = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return text.strip() or "No answer detected"


# ── Groq LLM evaluation ───────────────────────────────────────
def _groq_evaluate(question_text, model_answer, rubric, student_answer, max_marks, groq_api_key):
    from groq import Groq
    client = Groq(api_key=groq_api_key)
    prompt = f"""You are an expert teacher evaluating a student's handwritten answer.

Question: {question_text}
Model Answer: {model_answer}
Rubric: {json.dumps(rubric, indent=2)}
Student's Answer: {student_answer}
Maximum Marks: {max_marks}

Respond ONLY with JSON:
{{
  "score": <0 to {max_marks}>,
  "confidence": <0.0 to 1.0>,
  "matched_points": [<covered rubric points>],
  "missing_points": [<missed rubric points>],
  "feedback": "<1-2 sentence feedback>",
  "ocr_quality_concern": <true or false>
}}"""
    resp   = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )
    result = json.loads(resp.choices[0].message.content)
    result["score"] = max(0.0, min(float(result.get("score", 0)), max_marks))
    return result


# ── Local fallback evaluation ─────────────────────────────────
def _local_evaluate(model_answer, rubric, student_answer, max_marks):
    import re
    stop = {"the","a","an","is","are","was","were","be","been","have","has","had",
            "do","does","did","will","would","could","should","to","of","in","on",
            "at","by","for","with","and","or","but","it","its","this","that","not"}

    def keywords(text):
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        return [w for w in words if w not in stop]

    student_lower  = student_answer.lower().strip()
    model_keywords = keywords(model_answer)

    rubric_list = rubric if isinstance(rubric, list) else \
                  list(rubric.values()) if isinstance(rubric, dict) else [str(rubric)]

    matched, missing = [], []
    for point in rubric_list:
        pt_kw = keywords(str(point))
        if pt_kw and sum(1 for k in pt_kw if k in student_lower) / len(pt_kw) >= 0.4:
            matched.append(point)
        else:
            missing.append(point)

    rubric_score   = len(matched) / len(rubric_list) if rubric_list else 0
    kw_hits        = sum(1 for k in model_keywords if k in student_lower)
    keyword_score  = min(kw_hits / len(model_keywords), 1.0) if model_keywords else 0.5
    word_count     = len(student_lower.split())
    length_factor  = min(word_count / 10, 1.0)
    combined       = rubric_score * 0.6 + keyword_score * 0.3 + length_factor * 0.1
    score          = round(max(0, min(combined * max_marks, max_marks)), 1)
    pct            = score / max_marks * 100 if max_marks else 0

    if pct >= 80:   fb = "Excellent answer!"
    elif pct >= 60: fb = "Good attempt. Most key points covered."
    elif pct >= 40: fb = "Partial answer. Some important points missing."
    else:           fb = "Answer needs improvement."
    if missing:
        fb += f" Missing: {', '.join(str(m).split(':')[0] for m in missing[:2])}."

    return {
        "score": score, "confidence": round(min(0.5 + keyword_score * 0.4 + length_factor * 0.1, 0.95), 2),
        "matched_points": matched, "missing_points": missing,
        "feedback": fb, "ocr_quality_concern": word_count < 3
    }


# ── Main entry point called by ocr_tasks.py ──────────────────
def process_submission(submission_id, image_path, questions, bounding_boxes, groq_api_key=""):
    """
    Parameters
    ----------
    submission_id  : int
    image_path     : str  — path to uploaded image/PDF page
    questions      : list of dicts with keys:
                     question_number, question_text, model_answer, max_marks, rubric, question_id
    bounding_boxes : list of dicts with keys: question_number, x, y, width, height
    groq_api_key   : str

    Returns
    -------
    {"status": "success", "results": [...]}  or  {"status": "error", "message": "..."}
    """
    try:
        # ── Load image ───────────────────────────────────────
        ext = os.path.splitext(image_path)[1].lower()
        if ext == ".pdf":
            import fitz  # PyMuPDF
            doc  = fitz.open(image_path)
            page = doc[0]
            pix  = page.get_pixmap(dpi=150)
            arr  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            image_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if pix.n == 3 else \
                        cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        else:
            image_bgr = cv2.imread(image_path)

        if image_bgr is None:
            return {"status": "error", "message": f"Cannot read image: {image_path}"}

        # ── ArUco warp ───────────────────────────────────────
        warped, aligned = _detect_and_warp(image_bgr)
        if not aligned:
            print(f"ArUco markers not found for submission {submission_id}, using section-split fallback.")
            warped = image_bgr

        page_h = warped.shape[0]
        page_w = warped.shape[1]

        # Build bbox lookup
        bbox_map = {b["question_number"]: b for b in bounding_boxes}

        results = []
        num_questions = len(questions)
        for idx, q in enumerate(questions):
            q_num = q["question_number"]
            bbox  = bbox_map.get(q_num)

            # ── OCR ──────────────────────────────────────────
            if aligned and bbox:
                # ArUco aligned — use exact bounding box crop
                crop = _crop_region(warped, bbox, page_h)
            elif not aligned and num_questions > 1:
                # No ArUco — split image equally into sections per question
                section_h = page_h // num_questions
                y1 = idx * section_h
                y2 = (idx + 1) * section_h if idx < num_questions - 1 else page_h
                crop = warped[y1:y2, 0:page_w]
            else:
                # Single question or last resort — use full image
                crop = warped

            try:
                extracted_text = _ocr_crop(crop)
            except Exception as e:
                print(f"TrOCR failed for Q{q_num}: {e}")
                extracted_text = "OCR failed"

            # ── Evaluate ─────────────────────────────────────
            try:
                if groq_api_key and groq_api_key.strip():
                    eval_result = _groq_evaluate(
                        question_text=q["question_text"],
                        model_answer=q["model_answer"],
                        rubric=q["rubric"],
                        student_answer=extracted_text,
                        max_marks=q["max_marks"],
                        groq_api_key=groq_api_key
                    )
                else:
                    eval_result = _local_evaluate(
                        model_answer=q["model_answer"],
                        rubric=q["rubric"],
                        student_answer=extracted_text,
                        max_marks=q["max_marks"]
                    )
            except Exception as e:
                print(f"Evaluation failed for Q{q_num}: {e}, using local fallback.")
                eval_result = _local_evaluate(
                    model_answer=q["model_answer"],
                    rubric=q["rubric"],
                    student_answer=extracted_text,
                    max_marks=q["max_marks"]
                )

            results.append({
                "question_number": q_num,
                "question_id":     q["question_id"],
                "extracted_text":  extracted_text,
                "score":           eval_result["score"],
                "confidence":      eval_result["confidence"],
                "feedback":        eval_result["feedback"],
                "matched_points":  eval_result.get("matched_points", []),
                "missing_points":  eval_result.get("missing_points", []),
                "ocr_quality_concern": eval_result.get("ocr_quality_concern", False)
            })

        return {"status": "success", "results": results}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

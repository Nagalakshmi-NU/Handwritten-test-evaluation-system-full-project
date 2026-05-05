"""
ml_pipeline.py
Full pipeline: ArUco alignment → answer-region crop → Groq Vision OCR → Groq LLM evaluation
Uses Groq Vision instead of TrOCR to avoid RAM issues on Render free tier.
"""

import os, json, cv2, numpy as np, base64
from PIL import Image


# ── ArUco detection & perspective correction ─────────────────
def _detect_and_warp(image_bgr):
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
            id_to_corner[mid] = c.mean(axis=0)

    if len(id_to_corner) < 4:
        return image_bgr, False

    src = np.float32([id_to_corner[0], id_to_corner[1],
                      id_to_corner[2], id_to_corner[3]])
    W, H = 794, 1123
    dst  = np.float32([[0, 0], [W, 0], [0, H], [W, H]])
    M    = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image_bgr, M, (W, H))
    return warped, True


# ── Crop answer region from bounding box ─────────────────────
def _crop_region(image_bgr, bbox, page_h):
    x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["width"]), int(bbox["height"])
    img_h, img_w = image_bgr.shape[:2]
    scale_x = img_w / 794
    scale_y = img_h / 1123
    x  = max(0, int(x * scale_x))
    y  = max(0, int(y * scale_y))
    x2 = min(img_w, x + int(w * scale_x))
    y2 = min(img_h, y + int(h * scale_y))
    crop = image_bgr[y:y2, x:x2]
    return crop if crop.size > 0 else image_bgr


# ── Convert image to base64 for Groq Vision ──────────────────
def _image_to_base64(image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    # Resize if too large to save tokens
    max_size = 1024
    if pil.width > max_size or pil.height > max_size:
        pil.thumbnail((max_size, max_size), Image.LANCZOS)
    import io
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Groq Vision OCR ───────────────────────────────────────────
def _groq_ocr(crop_bgr, groq_api_key):
    """Use Groq Vision to extract handwritten text from image crop."""
    try:
        from groq import Groq
        client = Groq(api_key=groq_api_key)
        img_b64 = _image_to_base64(crop_bgr)
        response = client.chat.completions.create(
            model="llama-4-scout-17b-16e-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    },
                    {
                        "type": "text",
                        "text": "This is a handwritten student answer. Extract and transcribe ONLY the handwritten text exactly as written. Do not add any explanation or commentary. If no text is visible, respond with 'No answer written'."
                    }
                ]
            }],
            max_tokens=512,
            temperature=0
        )
        text = response.choices[0].message.content.strip()
        return text if text else "No answer detected"
    except Exception as e:
        print(f"Groq Vision OCR failed: {e}")
        return "OCR failed"


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
    resp = client.chat.completions.create(
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

    rubric_score  = len(matched) / len(rubric_list) if rubric_list else 0
    kw_hits       = sum(1 for k in model_keywords if k in student_lower)
    keyword_score = min(kw_hits / len(model_keywords), 1.0) if model_keywords else 0.5
    word_count    = len(student_lower.split())
    length_factor = min(word_count / 10, 1.0)
    combined      = rubric_score * 0.6 + keyword_score * 0.3 + length_factor * 0.1
    score         = round(max(0, min(combined * max_marks, max_marks)), 1)
    pct           = score / max_marks * 100 if max_marks else 0

    if pct >= 80:   fb = "Excellent answer!"
    elif pct >= 60: fb = "Good attempt. Most key points covered."
    elif pct >= 40: fb = "Partial answer. Some important points missing."
    else:           fb = "Answer needs improvement."
    if missing:
        fb += f" Missing: {', '.join(str(m).split(':')[0] for m in missing[:2])}."

    return {
        "score": score,
        "confidence": round(min(0.5 + keyword_score * 0.4 + length_factor * 0.1, 0.95), 2),
        "matched_points": matched, "missing_points": missing,
        "feedback": fb, "ocr_quality_concern": word_count < 3
    }


# ── Main entry point ──────────────────────────────────────────
def process_submission(submission_id, image_path, questions, bounding_boxes, groq_api_key=""):
    try:
        # Load image
        ext = os.path.splitext(image_path)[1].lower()
        if ext == ".pdf":
            import fitz
            doc  = fitz.open(image_path)
            pix  = doc[0].get_pixmap(dpi=150)
            arr  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            image_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if pix.n == 3 else \
                        cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        else:
            image_bgr = cv2.imread(image_path)

        if image_bgr is None:
            return {"status": "error", "message": f"Cannot read image: {image_path}"}

        # ArUco warp
        warped, aligned = _detect_and_warp(image_bgr)
        if not aligned:
            print(f"ArUco not found for submission {submission_id}, using section-split.")
            warped = image_bgr

        page_h = warped.shape[0]
        page_w = warped.shape[1]
        bbox_map = {b["question_number"]: b for b in bounding_boxes}
        num_questions = len(questions)
        results = []

        for idx, q in enumerate(questions):
            q_num = q["question_number"]
            bbox  = bbox_map.get(q_num)

            # Crop answer region
            if aligned and bbox:
                crop = _crop_region(warped, bbox, page_h)
            elif not aligned and num_questions > 1:
                section_h = page_h // num_questions
                y1 = idx * section_h
                y2 = (idx + 1) * section_h if idx < num_questions - 1 else page_h
                crop = warped[y1:y2, 0:page_w]
            else:
                crop = warped

            # OCR using Groq Vision (no RAM needed)
            if groq_api_key and groq_api_key.strip():
                extracted_text = _groq_ocr(crop, groq_api_key)
            else:
                extracted_text = "No answer detected (API key missing)"

            # Evaluate
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
                print(f"Evaluation failed for Q{q_num}: {e}")
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

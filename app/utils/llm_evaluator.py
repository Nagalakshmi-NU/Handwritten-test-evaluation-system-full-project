import json
import re

def evaluate_answer(question_text, model_answer, rubric, student_answer, max_marks, groq_api_key=""):
    """
    Local rule-based evaluation — no API needed.
    Compares student answer against model answer and rubric using keyword matching.
    """

    # If Groq API key is provided, use it
    if groq_api_key and groq_api_key.strip():
        try:
            return _groq_evaluate(question_text, model_answer, rubric, student_answer, max_marks, groq_api_key)
        except Exception as e:
            print(f"Groq failed, using local evaluation: {e}")

    # Local evaluation
    return _local_evaluate(model_answer, rubric, student_answer, max_marks)


def _local_evaluate(model_answer, rubric, student_answer, max_marks):
    """Keyword-based local scoring."""

    student_lower  = student_answer.lower().strip()
    model_lower    = model_answer.lower().strip()

    if not student_lower or student_lower in ["no answer provided", "no answer detected"]:
        return {
            "score": 0, "confidence": 0.9,
            "matched_points": [], "missing_points": _extract_rubric_points(rubric),
            "feedback": "No answer was provided for this question.",
            "ocr_quality_concern": False
        }

    # Extract keywords from model answer
    model_keywords = _extract_keywords(model_lower)
    student_keywords = _extract_keywords(student_lower)

    # Check rubric points
    matched, missing = [], []
    rubric_points = _extract_rubric_points(rubric)

    for point in rubric_points:
        point_keywords = _extract_keywords(point.lower())
        # Check if at least 40% of point keywords appear in student answer
        if point_keywords:
            hits = sum(1 for kw in point_keywords if kw in student_lower)
            if hits / len(point_keywords) >= 0.4:
                matched.append(point)
            else:
                missing.append(point)
        else:
            missing.append(point)

    # Score based on rubric coverage
    rubric_score = (len(matched) / len(rubric_points)) if rubric_points else 0

    # Score based on keyword overlap with model answer
    if model_keywords:
        keyword_hits = sum(1 for kw in model_keywords if kw in student_lower)
        keyword_score = min(keyword_hits / len(model_keywords), 1.0)
    else:
        keyword_score = 0.5

    # Length penalty — very short answers get penalized
    word_count = len(student_lower.split())
    length_factor = min(word_count / 10, 1.0)

    # Combined score (rubric 60%, keywords 30%, length 10%)
    combined = (rubric_score * 0.6) + (keyword_score * 0.3) + (length_factor * 0.1)
    raw_score = round(combined * max_marks, 1)
    raw_score = max(0, min(raw_score, max_marks))

    # Confidence based on answer length and keyword overlap
    confidence = round(min(0.5 + (keyword_score * 0.4) + (length_factor * 0.1), 0.95), 2)

    # Generate feedback
    feedback = _generate_feedback(matched, missing, raw_score, max_marks, word_count)

    return {
        "score": raw_score,
        "confidence": confidence,
        "matched_points": matched,
        "missing_points": missing,
        "feedback": feedback,
        "ocr_quality_concern": word_count < 3
    }


def _extract_keywords(text):
    """Extract meaningful keywords, ignoring stop words."""
    stop_words = {
        "the","a","an","is","are","was","were","be","been","being","have","has","had",
        "do","does","did","will","would","could","should","may","might","shall","can",
        "to","of","in","on","at","by","for","with","about","as","into","through",
        "and","or","but","if","then","that","this","it","its","their","they","we",
        "he","she","i","you","my","your","our","his","her","which","what","when",
        "where","how","why","not","no","so","also","both","each","more","most","other"
    }
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    return [w for w in words if w not in stop_words]


def _extract_rubric_points(rubric):
    """Extract rubric points as list of strings."""
    if isinstance(rubric, list):
        return [str(r) for r in rubric]
    if isinstance(rubric, dict):
        return [str(v) for v in rubric.values()]
    if isinstance(rubric, str):
        try:
            parsed = json.loads(rubric)
            return _extract_rubric_points(parsed)
        except:
            return [rubric]
    return []


def _generate_feedback(matched, missing, score, max_marks, word_count):
    """Generate human-readable feedback."""
    pct = (score / max_marks * 100) if max_marks else 0

    if word_count < 3:
        return "Answer appears to be missing or too short. Please write a complete answer."

    if pct >= 80:
        base = "Excellent answer! You covered the key concepts well."
    elif pct >= 60:
        base = "Good attempt. Most key points are covered."
    elif pct >= 40:
        base = "Partial answer. Some important points are missing."
    else:
        base = "Answer needs improvement. Key concepts are not clearly explained."

    if missing:
        missing_short = [m.split(':')[0] for m in missing[:2]]
        base += f" Missing: {', '.join(missing_short)}."

    if matched:
        base += f" Covered {len(matched)}/{len(matched)+len(missing)} rubric points."

    return base


def _groq_evaluate(question_text, model_answer, rubric, student_answer, max_marks, groq_api_key):
    """Groq LLM evaluation (used only if API key is provided)."""
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
  "confidence": <0 to 1>,
  "matched_points": [<covered rubric points>],
  "missing_points": [<missed rubric points>],
  "feedback": "<1-2 sentence feedback>",
  "ocr_quality_concern": <true or false>
}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )
    result = json.loads(response.choices[0].message.content)
    result["score"] = max(0, min(float(result.get("score", 0)), max_marks))
    return result

import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, A3, landscape
from reportlab.lib import colors
import os

ARUCO_DICTS = {
    "DICT_4X4_50":   cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100":  cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50":   cv2.aruco.DICT_5X5_50,
    "DICT_6X6_50":   cv2.aruco.DICT_6X6_50,
}

def _page_size(size_name, orientation):
    base = A3 if size_name == "A3" else A4
    return landscape(base) if orientation == "landscape" else base

def _save_aruco(marker_id, path, size_px, dict_name):
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS.get(dict_name, cv2.aruco.DICT_4X4_50))
    img = cv2.aruco.generateImageMarker(d, marker_id, size_px)
    cv2.imwrite(path, img)

def generate_answer_sheet(test_id, title, questions, output_path,
                           page_size="A4", orientation="portrait",
                           margin=40, aruco_size=30,
                           aruco_dict="DICT_4X4_50", aruco_start_id=0):
    PAGE_W, PAGE_H = _page_size(page_size, orientation)
    HEADER_H = 80

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Generate ArUco markers
    aruco_paths = []
    for i in range(4):
        path = output_path.replace(".pdf", f"_aruco_{i}.png")
        _save_aruco(aruco_start_id + i, path, 100, aruco_dict)
        aruco_paths.append(path)

    c = canvas.Canvas(output_path, pagesize=(PAGE_W, PAGE_H))

    def draw_markers():
        positions = [
            (margin, PAGE_H - margin - aruco_size),
            (PAGE_W - margin - aruco_size, PAGE_H - margin - aruco_size),
            (margin, margin),
            (PAGE_W - margin - aruco_size, margin),
        ]
        for i, (x, y) in enumerate(positions):
            c.drawImage(aruco_paths[i], x, y, width=aruco_size, height=aruco_size)

    def draw_header():
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(PAGE_W / 2, PAGE_H - margin - 15, title)
        c.setFont("Helvetica", 10)
        c.drawString(margin, PAGE_H - margin - 35, f"Test ID: {test_id}")
        c.drawString(margin, PAGE_H - margin - 50, "Student Name: ___________________________")
        c.drawString(PAGE_W - 200, PAGE_H - margin - 50, "Roll No: ____________")
        c.setStrokeColor(colors.black)
        c.line(margin, PAGE_H - margin - 60, PAGE_W - margin, PAGE_H - margin - 60)

    draw_markers()
    draw_header()

    bounding_boxes = []
    current_y = PAGE_H - margin - HEADER_H
    box_width  = PAGE_W - 2 * margin

    for q in questions:
        q_num     = q["question_number"]
        q_text    = q.get("question_text", f"Question {q_num}")
        max_marks = q["max_marks"]
        box_h     = q.get("box_height", 120)

        if current_y - box_h - 30 < margin + aruco_size + 10:
            c.showPage()
            draw_markers()
            current_y = PAGE_H - margin - 20

        c.setFont("Helvetica-Bold", 10)
        c.drawString(margin, current_y - 12, f"Q{q_num}. {q_text}  [{max_marks} marks]")

        box_y = current_y - 15 - box_h
        c.setStrokeColor(colors.black)
        c.setLineWidth(0.5)
        c.rect(margin, box_y, box_width, box_h)

        bounding_boxes.append({
            "question_number": q_num,
            "x": int(margin),
            "y": int(PAGE_H - (box_y + box_h)),
            "width": int(box_width),
            "height": int(box_h)
        })
        current_y = box_y - 15

    c.save()
    for p in aruco_paths:
        if os.path.exists(p):
            os.remove(p)

    return bounding_boxes

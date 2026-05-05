# AutoGrade — AI Handwritten Test Evaluation System

An AI-powered web application that evaluates handwritten answer sheets using OCR and LLM.

## Project Structure

```
AutoGrade/
├── app/                        # Flask backend
│   ├── routes/                 # API route handlers
│   │   ├── auth.py             # Login, register, password reset
│   │   ├── teacher.py          # Test creation, review, publish
│   │   ├── student.py          # Submit answers, view results
│   │   └── admin.py            # User management, analytics
│   ├── tasks/
│   │   └── ocr_tasks.py        # Async evaluation pipeline
│   ├── utils/
│   │   ├── llm_evaluator.py    # Groq LLM + local evaluation
│   │   └── pdf_generator.py    # ArUco PDF answer sheet generator
│   ├── models.py               # Database models
│   └── __init__.py             # Flask app factory
├── frontend/                   # HTML/CSS/JS frontend
│   └── static/
│       ├── css/                # Stylesheets
│       ├── js/                 # JavaScript files
│       └── pages/              # 16 HTML pages
├── uploads/                    # Student uploaded answer sheets
├── generated_pdfs/             # Generated ArUco answer sheet PDFs
├── ml_pipeline.py              # ArUco + TrOCR + Groq ML pipeline
├── config.py                   # App configuration
├── run.py                      # App entry point
├── requirements.txt            # Python dependencies
└── .env                        # Environment variables (not in git)
```

## Setup & Run

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Configure .env
```
SECRET_KEY=your-secret-key
DATABASE_URL=postgresql://postgres:password@localhost:5432/testpaper_db
GROQ_API_KEY=your-groq-api-key
```

### 3. Run
```
python run.py
```

Open http://127.0.0.1:5000

## Tech Stack
- **Backend:** Flask, PostgreSQL, SQLAlchemy, Flask-Login
- **ML:** TrOCR (handwriting OCR), OpenCV (ArUco detection)
- **AI:** Groq LLM (llama-3.3-70b-versatile)
- **Frontend:** HTML, CSS, JavaScript
- **Async:** Celery + Redis (falls back to threading)

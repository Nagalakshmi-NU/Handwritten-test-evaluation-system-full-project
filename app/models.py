from app import db, login_manager
from flask_login import UserMixin
from datetime import datetime

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(UserMixin, db.Model):
    __tablename__ = "users"
    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(100), nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role          = db.Column(db.String(20), nullable=False)
    admin_id      = db.Column(db.String(50), nullable=True)
    employee_id   = db.Column(db.String(50), nullable=True)
    student_id    = db.Column(db.String(50), nullable=True)
    subjects      = db.Column(db.String(300), nullable=True)
    status        = db.Column(db.String(20), default='approved')
    last_login    = db.Column(db.DateTime, nullable=True)
    login_attempts= db.Column(db.Integer, default=0)
    locked_until  = db.Column(db.DateTime, nullable=True)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

class Test(db.Model):
    __tablename__ = "tests"
    id         = db.Column(db.Integer, primary_key=True)
    teacher_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    title      = db.Column(db.String(200), nullable=False)
    subject    = db.Column(db.String(100), nullable=True)
    total_marks= db.Column(db.Integer, nullable=False)
    status     = db.Column(db.String(20), default='active')
    deadline   = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Question(db.Model):
    __tablename__ = "questions"
    id              = db.Column(db.Integer, primary_key=True)
    test_id         = db.Column(db.Integer, db.ForeignKey("tests.id"), nullable=False)
    question_number = db.Column(db.Integer, nullable=False)
    question_text   = db.Column(db.Text, nullable=True)
    model_answer    = db.Column(db.Text, nullable=False)
    max_marks       = db.Column(db.Integer, nullable=False)
    rubric_json     = db.Column(db.Text, nullable=False)

class QuestionBank(db.Model):
    __tablename__ = "question_bank"
    id            = db.Column(db.Integer, primary_key=True)
    teacher_id    = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    subject       = db.Column(db.String(100), nullable=True)
    question_text = db.Column(db.Text, nullable=False)
    model_answer  = db.Column(db.Text, nullable=False)
    max_marks     = db.Column(db.Integer, nullable=False)
    rubric_json   = db.Column(db.Text, nullable=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

class Submission(db.Model):
    __tablename__ = "submissions"
    id          = db.Column(db.Integer, primary_key=True)
    test_id     = db.Column(db.Integer, db.ForeignKey("tests.id"), nullable=False)
    student_id  = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    status      = db.Column(db.String(50), default="pending")

class SubmissionPage(db.Model):
    __tablename__ = "submission_pages"
    id            = db.Column(db.Integer, primary_key=True)
    submission_id = db.Column(db.Integer, db.ForeignKey("submissions.id"), nullable=False)
    page_number   = db.Column(db.Integer, nullable=False)
    image_path    = db.Column(db.String(300), nullable=False)
    processed_text= db.Column(db.Text)

class Evaluation(db.Model):
    __tablename__ = "evaluations"
    id            = db.Column(db.Integer, primary_key=True)
    submission_id = db.Column(db.Integer, db.ForeignKey("submissions.id"), nullable=False)
    question_id   = db.Column(db.Integer, db.ForeignKey("questions.id"), nullable=False)
    extracted_answer = db.Column(db.Text)
    ai_score      = db.Column(db.Float)
    ai_feedback   = db.Column(db.Text)
    teacher_score = db.Column(db.Float)
    final_score   = db.Column(db.Float)
    confidence    = db.Column(db.Float)

class ActivityLog(db.Model):
    __tablename__ = "activity_logs"
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    action     = db.Column(db.String(200), nullable=False)
    timestamp  = db.Column(db.DateTime, default=datetime.utcnow)

class TeacherFeedback(db.Model):
    __tablename__ = "teacher_feedback"
    id         = db.Column(db.Integer, primary_key=True)
    teacher_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    message    = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Notification(db.Model):
    __tablename__ = "notifications"
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    message    = db.Column(db.String(300), nullable=False)
    is_read    = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Announcement(db.Model):
    __tablename__ = "announcements"
    id         = db.Column(db.Integer, primary_key=True)
    teacher_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    test_id    = db.Column(db.Integer, db.ForeignKey("tests.id"), nullable=True)
    message    = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

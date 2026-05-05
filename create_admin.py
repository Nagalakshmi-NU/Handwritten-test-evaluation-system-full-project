"""
Run this once after Render deployment to create the admin account.
Usage: python create_admin.py
"""
import os
from dotenv import load_dotenv
load_dotenv()

db_url = input("Paste your Render External Database URL here: ").strip()
os.environ["DATABASE_URL"] = db_url

from app import create_app, db
from app.models import User
from werkzeug.security import generate_password_hash

app = create_app()

with app.app_context():
    db.create_all()
    existing = User.query.filter_by(role="admin").first()
    if existing:
        print(f"Admin already exists: {existing.email}")
    else:
        admin = User(
            name="Admin",
            email="admin@autograde.com",
            password_hash=generate_password_hash("admin123"),
            role="admin",
            status="approved",
            admin_id="ADM001"
        )
        db.session.add(admin)
        db.session.commit()
        print("========================================")
        print("Admin account created!")
        print("Email:    admin@autograde.com")
        print("Password: admin123")
        print("========================================")

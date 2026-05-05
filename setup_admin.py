import os
import sys

db_url = "postgresql://autograde_db_user:eO3IiZS1p4oJtOTxx4bQBEzQJzq5Qp5y@dpg-d7sp70lckfvc73cl0fo0-a.singapore-postgres.render.com/autograde_db"
os.environ["DATABASE_URL"] = db_url
os.environ["SECRET_KEY"] = "autograde-super-secret-key-2025"
os.environ["GROQ_API_KEY"] = ""

print("Connecting to database...")

from app import create_app, db
from app.models import User
from werkzeug.security import generate_password_hash

app = create_app()

with app.app_context():
    print("Creating tables...")
    db.create_all()
    print("Tables created!")

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

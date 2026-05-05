from app import create_app, db
import os

app = create_app()

try:
    with app.app_context():
        db.create_all()
        print("Tables created successfully!")
except Exception as e:
    print(f"DB init warning: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("RAILWAY_ENVIRONMENT") is None
    app.run(debug=debug, host='0.0.0.0', port=port)
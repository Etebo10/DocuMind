from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(320), unique=True, nullable=False)
    name = db.Column(db.String(120))
    password_hash = db.Column(db.String(200))
    oauth_provider = db.Column(db.String(50))
    oauth_id = db.Column(db.String(200))
    # Security fields
    failed_attempts = db.Column(db.Integer, default=0, nullable=False)
    last_failed_at = db.Column(db.DateTime)
    locked_until = db.Column(db.DateTime)
    email_verified = db.Column(db.Boolean, default=False, nullable=False)
    verification_sent_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<User {self.email}>'

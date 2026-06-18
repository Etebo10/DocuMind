import os
from datetime import datetime, timedelta
from flask import Flask, render_template, redirect, url_for, flash, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_wtf import CSRFProtect
from authlib.integrations.flask_client import OAuth
from models import db, User
from forms import SignupForm, LoginForm
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_mail import Mail, Message
from flask_talisman import Talisman
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature

load_dotenv()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///data.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    # Secure cookie flags for production
    app.config['SESSION_COOKIE_SECURE'] = os.environ.get('SESSION_COOKIE_SECURE', 'False') == 'True'
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

    # Mail config (optional)
    app.config['MAIL_SERVER'] = os.environ.get('SMTP_HOST')
    app.config['MAIL_PORT'] = int(os.environ.get('SMTP_PORT', 0)) if os.environ.get('SMTP_PORT') else None
    app.config['MAIL_USERNAME'] = os.environ.get('SMTP_USER')
    app.config['MAIL_PASSWORD'] = os.environ.get('SMTP_PASS')
    app.config['MAIL_USE_TLS'] = os.environ.get('SMTP_USE_TLS', 'True') == 'True'
    app.config['MAIL_USE_SSL'] = os.environ.get('SMTP_USE_SSL', 'False') == 'True'

    db.init_app(app)
    login_manager = LoginManager()
    login_manager.login_view = 'auth'
    login_manager.init_app(app)
    csrf = CSRFProtect(app)

    # Security middlewares
    talisman = Talisman(app, content_security_policy=None)

    # Rate limiter
    limiter = Limiter(key_func=get_remote_address)
    limiter.init_app(app)

    # Mail
    mail = Mail(app)

    oauth = OAuth(app)
    oauth.register(
        name='google',
        client_id=os.environ.get('GOOGLE_CLIENT_ID'),
        client_secret=os.environ.get('GOOGLE_CLIENT_SECRET'),
        access_token_url='https://oauth2.googleapis.com/token',
        access_token_params=None,
        authorize_url='https://accounts.google.com/o/oauth2/v2/auth',
        authorize_params=None,
        api_base_url='https://www.googleapis.com/oauth2/v1/',
        client_kwargs={'scope': 'openid email profile'},
    )

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Helper: serializer for email tokens
    serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])

    @app.context_processor
    def inject_year():
        return {'current_year': datetime.utcnow().year}

    @app.before_first_request
    def create_tables():
        db.create_all()

    @app.route('/')
    def index():
        return render_template('index.html', user=current_user)

    @app.route('/auth', methods=['GET', 'POST'])
    @limiter.limit("10/minute")
    def auth():
        signup = SignupForm(prefix='su')
        login = LoginForm(prefix='li')

        # Signup flow
        if signup.validate_on_submit() and signup.submit.data:
            if User.query.filter_by(email=signup.email.data.lower()).first():
                flash('Email already registered', 'danger')
            else:
                pw = generate_password_hash(signup.password.data)
                user = User(email=signup.email.data.lower(), name=signup.name.data, password_hash=pw)
                db.session.add(user)
                user.verification_sent_at = datetime.utcnow()
                db.session.commit()
                # send verification email (if configured)
                try:
                    token = serializer.dumps(user.email, salt='email-verify')
                    verify_url = url_for('verify_email', token=token, _external=True)
                    if app.config.get('MAIL_SERVER'):
                        msg = Message('Verify your DocuMind email', recipients=[user.email])
                        msg.body = f'Click to verify your email: {verify_url}'
                        mail.send(msg)
                    else:
                        # Development fallback: show link in flash
                        flash(f'Email verification link (dev): {verify_url}', 'info')
                except Exception:
                    flash('Unable to send verification email right now', 'warning')

                login_user(user)
                flash('Welcome — account created. Please verify your email.', 'success')
                return redirect(url_for('index'))

        # Login flow with account lockout
        if login.validate_on_submit() and login.submit.data:
            user = User.query.filter_by(email=login.email.data.lower()).first()
            now = datetime.utcnow()
            if user:
                if user.locked_until and user.locked_until > now:
                    remaining = int((user.locked_until - now).total_seconds() / 60) + 1
                    flash(f'Account locked due to failed attempts. Try again in {remaining} minutes.', 'danger')
                    return render_template('auth.html', signup=signup, login=login, user=current_user)

                if user and check_password_hash(user.password_hash, login.password.data):
                    user.failed_attempts = 0
                    user.locked_until = None
                    db.session.commit()
                    login_user(user, remember=login.remember.data)
                    flash('Signed in successfully', 'success')
                    return redirect(url_for('index'))

                # failed auth
                if user:
                    user.failed_attempts = (user.failed_attempts or 0) + 1
                    user.last_failed_at = now
                    if user.failed_attempts >= 5:
                        user.locked_until = now + timedelta(minutes=15)
                    db.session.commit()

            flash('Invalid credentials', 'danger')

        return render_template('auth.html', signup=signup, login=login, user=current_user)

    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        flash('Signed out', 'info')
        return redirect(url_for('index'))

    @app.route('/login/google')
    @limiter.limit("6/minute")
    def login_google():
        redirect_uri = url_for('authorize_google', _external=True)
        return oauth.google.authorize_redirect(redirect_uri)

    @app.route('/authorize/google')
    @limiter.limit("6/minute")
    def authorize_google():
        token = oauth.google.authorize_access_token()
        userinfo = oauth.google.get('userinfo').json()
        email = userinfo.get('email')
        sub = userinfo.get('id')
        name = userinfo.get('name') or email

        user = User.query.filter_by(email=email).first()
        if not user:
            user = User(email=email, name=name, oauth_provider='google', oauth_id=sub, email_verified=True)
            db.session.add(user)
            db.session.commit()
        login_user(user)
        flash('Signed in with Google', 'success')
        return redirect(url_for('index'))

    # Email verification route
    @app.route('/verify-email/<token>')
    def verify_email(token):
        try:
            email = serializer.loads(token, salt='email-verify', max_age=60*60*24)
        except SignatureExpired:
            flash('Verification link expired', 'danger')
            return redirect(url_for('auth'))
        except BadSignature:
            flash('Invalid verification token', 'danger')
            return redirect(url_for('auth'))

        user = User.query.filter_by(email=email).first()
        if user:
            user.email_verified = True
            db.session.commit()
            flash('Email verified — thank you!', 'success')
        return redirect(url_for('index'))

    @app.route('/resend-verification')
    @login_required
    def resend_verification():
        user = current_user
        if user.email_verified:
            flash('Your email is already verified.', 'info')
            return redirect(url_for('index'))
        token = serializer.dumps(user.email, salt='email-verify')
        verify_url = url_for('verify_email', token=token, _external=True)
        if app.config.get('MAIL_SERVER'):
            try:
                msg = Message('Verify your DocuMind email', recipients=[user.email])
                msg.body = f'Click to verify your email: {verify_url}'
                mail.send(msg)
                flash('Verification email sent', 'success')
            except Exception:
                flash('Failed to send email', 'warning')
        else:
            flash(f'Email verification link (dev): {verify_url}', 'info')
        return redirect(url_for('index'))

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)

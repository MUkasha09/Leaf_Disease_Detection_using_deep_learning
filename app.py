from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import numpy as np
import json
import uuid
import tensorflow as tf
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = os.urandom(24)  # Use a cryptographically secure random secret key
app.config['UPLOAD_FOLDER'] = "static/uploads"
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'  # Redirect users to login page if not authenticated

# User Model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)

# History Model
class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    disease_name = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

# Initialize the database
def init_db():
    with app.app_context():
        db.create_all()

# Flask-Login User Loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Load AI model with error handling
def load_model():
    try:
        # Use a relative path or environment variable for model path
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'plant_disease_recog_model_pwp.keras')
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load disease labels with error handling
def load_disease_labels():
    try:
        # Use a relative path or environment variable for labels path
        labels_path = os.path.join(os.path.dirname(__file__), 'plant_disease.json')
        with open(labels_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading disease labels: {e}")
        return {}

# Load model and labels
model = load_model()
plant_disease = load_disease_labels()

# Function to extract features
def extract_features(image_path):
    try:
        image = tf.keras.utils.load_img(image_path, target_size=(160, 160))
        image_array = tf.keras.utils.img_to_array(image)
        return np.array([image_array])
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# Model prediction function
def model_predict(image_path):
    if not model or not plant_disease:
        return "Model or labels not loaded"
    
    img = extract_features(image_path)
    if img is None:
        return "Image processing error"
    
    try:
        prediction = model.predict(img)
        predicted_label = plant_disease[prediction.argmax()]
        return predicted_label
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Prediction failed"

# ROUTES

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('upload'))
        else:
            flash('Invalid email or password', 'error')

    return render_template('login.html')

@app.route('/signup/', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Validate input
        if not username or not email or not password:
            flash('All fields are required', 'error')
            return render_template('signup.html')
        
        # Check if user already exists
        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            flash('Username or email already exists', 'error')
            return render_template('signup.html')

        # Create new user
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password=hashed_password)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Signup successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred during signup', 'error')
            print(f"Signup error: {e}")

    return render_template('signup.html')

@app.route('/logout/')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))

@app.route('/upload/', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        # Check if file is present
        if 'img' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)

        image = request.files['img']
        
        # Check if filename is empty
        if image.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        # Check if file is allowed
        if image and allowed_file(image.filename):
            # Secure the filename
            unique_filename = f"{uuid.uuid4().hex}_{secure_filename(image.filename)}"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            try:
                # Save the file
                image.save(image_path)

                # Get the prediction
                prediction = model_predict(image_path)

                # Ensure only the name is passed
                disease_name = prediction['name'] if isinstance(prediction, dict) else prediction

                # Save history to the database
                new_history = History(
                    user_id=current_user.id, 
                    image_path=f"/static/uploads/{unique_filename}", 
                    disease_name=disease_name
                )
                db.session.add(new_history)
                db.session.commit()

                return render_template('results.html', 
                                       result=True, 
                                       imagepath=f"/static/uploads/{unique_filename}", 
                                       prediction=disease_name)
            
            except Exception as e:
                flash(f'Error processing image: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Invalid file type', 'error')
            return redirect(request.url)

    return render_template('upload.html')

@app.route('/history/')
@login_required
def history():
    user_history = History.query.filter_by(user_id=current_user.id).order_by(History.timestamp.desc()).all()
    return render_template('history.html', history=user_history)

# Initialize database when running the app
init_db()

if __name__ == "__main__":
    app.run(debug=False)  # Set to False in production
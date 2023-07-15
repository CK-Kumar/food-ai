from flask import Flask, request, jsonify, session
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask import render_template
from fastai.vision.all import *
from config import ApplicationConfig
from models import db, User
from flask import render_template
from pathlib import Path
import pathlib
from PIL import Image
from io import BytesIO
import base64

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)
app.config.from_object(ApplicationConfig)

bcrypt = Bcrypt(app)
db.init_app(app)

with app.app_context():
    db.create_all()

#Labeling function required for load_learner to work
def GetLabel(fileName):
  return fileName.split('-')[0]

learn = load_learner(Path('export.pkl')) #Import Model
cors = CORS(app) #Request will get blocked otherwise on Localhost


@app.route("/register", methods=["POST"])
def register_user():
    email = request.json.get("email")
    password = request.json.get("password")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        return jsonify({"error": "User already exists"}), 409

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(email=email, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "User registered successfully"}), 201



@app.route("/users", methods = ["GET"])
def get_all_user():
    users = User.query.all()  
    usernames = [{"email" : user.email, "id": user.id} for user in users]
    return jsonify({"usernames": usernames}), 200



@app.route("/delete", methods=["DELETE"])
def delete_user():
    id = request.json.get("id")
    user = User.query.get(id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    db.session.delete(user)
    db.session.commit()

    return jsonify({"message": "User deleted successfully"}), 200



@app.route('/predict', methods=['POST'])
def predict():
    base64_string = request.json.get('imageString')
    image_bytes = base64.b64decode(base64_string)
    img = PILImage.create(image_bytes)
    label, _, probs = learn.predict(img)
    prediction = f'{label} ({torch.max(probs).item()*100:.0f}%)'
    return jsonify({
        "label": label,
        "prediction":(torch.max(probs).item()*100)
    })
   

@app.route("/")
def index():
    return render_template("upload.html")

@app.route('/result')
def show_result():
    return render_template('result.html')

if __name__ == "__main__":
    app.run(debug=True)
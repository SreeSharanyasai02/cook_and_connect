from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import os, json
from ultralytics import YOLO

from datetime import datetime
from collections import Counter


# ================= INGREDIENT NORMALIZATION =================
INGREDIENT_MAP = {
    "bell pepper": "capsicum",
    "green chilli": "chilli",
    "red chilli": "chilli",
    "scallion": "onion",
    "spring onion": "onion",
    "coriander leaves": "coriander",
    "cilantro": "coriander",
    "eggplant": "brinjal"
}

def normalize_ingredients(ingredients):
    normalized = []
    for ing in ingredients:
        ing = ing.lower().strip()
        normalized.append(INGREDIENT_MAP.get(ing, ing))
    return list(set(normalized))

# ================= APP SETUP =================
app = Flask(__name__)
CORS(app)

app.secret_key = "cook_connect_secret_key"
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False
app.permanent_session_lifetime = 86400

# ================= DATABASE =================
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ================= UPLOADS =================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ================= MODELS =================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    profile_pic = db.Column(db.String(200), default="default.png")
    diet = db.Column(db.String(30), default="Veg 🌱")
    cuisines = db.Column(db.String(200), default="Indian 🇮🇳")

# ================= LOAD AI =================
yolo_model = None

def get_yolo_model():
    global yolo_model
    if yolo_model is None:
        print("Loading YOLO...")
        yolo_model = YOLO("yolov8n.pt")
        print("YOLO Loaded")
    return yolo_model




# ================= LOAD RECIPES =================
with open("recipes.json", "r", encoding="utf-8") as f:
    RECIPE_DB = json.load(f)

# ================= RECIPE MATCHING =================
def find_best_recipe(user_ingredients):
    best_recipe = None
    best_score = 0
    user_set = set(user_ingredients)

    for recipe in RECIPE_DB:
        recipe_set = set(normalize_ingredients(recipe.get("ingredients", [])))
        score = len(recipe_set & user_set)
        if score > best_score:
            best_score = score
            best_recipe = recipe

    return best_recipe

def find_all_possible_recipes(user_ingredients):
    user_set = set(normalize_ingredients(user_ingredients))
    matches = []

    difficulty_rank = {
        "Easy": 0,
        "Medium": 1,
        "Hard": 2
    }

    for recipe in RECIPE_DB:
        recipe_set = set(normalize_ingredients(recipe.get("ingredients", [])))
        common = recipe_set & user_set
        missing = recipe_set - user_set

        if len(recipe_set) == 0:
            continue

        match_ratio = len(common) / len(recipe_set)

        if match_ratio >= 0.6 and len(missing) <= 1:
            matches.append({
                "name": recipe.get("name"),
                "preview": recipe.get("preview", ""),
                "difficulty": recipe.get("difficulty"),
                "time": recipe.get("time"),
                "calories": recipe.get("calories"),
                "missing": list(missing),
                "missing_count": len(missing),
                "difficulty_rank": difficulty_rank.get(recipe.get("difficulty"), 3)
            })

    matches.sort(key=lambda x: (
        x["missing_count"],
        x["difficulty_rank"],
        x["time"] or 999
    ))

    return matches


# ================= ROUTES =================
@app.route('/')
def splash():
    return render_template("splash.html")

@app.route('/login-page')
def login_page():
    return render_template("login.html")

@app.route('/signup-page')
def signup_page():
    return render_template("signup.html")

@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template("home.html")

@app.route('/explore')
def explore():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template("explore.html")

@app.route('/account')
def account():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    user = User.query.get(session['user_id'])
    return render_template("account.html", user=user)

@app.route('/recipe')
def recipe_page():
    return render_template("recipe.html")

@app.route('/cooking')
def cooking():
    return render_template("cooking.html")

@app.route('/memories-page')
def memories_page():
    return render_template("memories.html")

@app.route('/analytics-page')
def analytics_page():
    return render_template("analytics.html")

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ================= AUTH =================
@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    if User.query.filter_by(email=data['email']).first():
        return jsonify({"error": "Email already exists"}), 400

    user = User(
        name=data['name'],
        email=data['email'],
        password=generate_password_hash(data['password'])
    )

    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "Signup successful"})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    user = User.query.filter_by(email=data['email']).first()

    if user and check_password_hash(user.password, data['password']):
        session.permanent = True
        session['user_id'] = user.id
        session['user_name'] = user.name
        return jsonify({"message": "Login success"})

    return jsonify({"error": "Invalid credentials"}), 401

# ================= YOLO DETECTION =================
@app.route('/detect-ingredients', methods=['POST'])
def detect_ingredients():
    if 'image' not in request.files:
        return jsonify({"ingredients": []})

    file = request.files['image']
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    results = get_yolo_model()(path, conf=0.01)
    detected = []

    for r in results:
        if r.boxes:
            for box in r.boxes:
                cls_id = int(box.cls)
                label = r.names.get(cls_id, "unknown")
                detected.append(label)

    return jsonify({
        "ingredients": normalize_ingredients(detected)
    })

# ================= DISH OPTIONS =================
@app.route('/get-dish-options', methods=['POST'])
def get_dish_options():
    data = request.json
    ingredients = normalize_ingredients(data.get("ingredients", []))

    dishes = find_all_possible_recipes(ingredients)

    if not dishes:
        dishes = [{
            "name": "Quick Kitchen Stir Fry",
            "difficulty": "Easy",
            "time": 15,
            "calories": 250,
            "missing": [],
            "ai_generated": True
        }]

    return jsonify({
        "count": len(dishes),
        "dishes": dishes
    })

# ================= GENERATE RECIPE =================
@app.route('/generate-recipe', methods=['POST'])
def generate_recipe():
    data = request.json
    ingredients = normalize_ingredients(data.get("ingredients", []))
    selected_dish = data.get("selected_dish")

    if selected_dish:
        recipe = next(
            (r for r in RECIPE_DB
             if r.get("name", "").strip().lower() == selected_dish.strip().lower()),
            None
        )
    else:
        recipe = find_best_recipe(ingredients)

    if not recipe:
        recipe = {
            "name": "Custom Dish",
            "ingredients": ingredients,
            "calories": 220,
            "difficulty": "Easy",
            "steps": [{"text": "Cook everything well", "time": 10}]
        }

    timed_steps = []
    for step in recipe.get("steps", []):
        if isinstance(step, dict):
            timed_steps.append({
                "description": step.get("text"),
                "time": step.get("time", 5)
            })

    return jsonify({
        "name": recipe.get("name"),
        "ingredients": recipe.get("ingredients"),
        "calories": recipe.get("calories"),
        "difficulty": recipe.get("difficulty"),
        "steps": timed_steps,
        "total_time": sum(s["time"] for s in timed_steps)
    })

# ================= SAVE MEMORY =================
# ================= SAVE MEMORY =================
@app.route("/save-memory", methods=["POST"])
def save_memory():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    name = request.form.get("name")
    calories = request.form.get("calories")
    ingredients = request.form.get("ingredients")
    note = request.form.get("note")

    if ingredients:
        ingredients = json.loads(ingredients)

    image = request.files.get("image")

    # 🔴 If no image uploaded → reject
    if not image or image.filename == "":
        return jsonify({"error": "Image is required"}), 400

    os.makedirs("static/uploads", exist_ok=True)

    # Make filename unique using timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{secure_filename(image.filename)}"
    filepath = os.path.join("static/uploads", filename)
    image.save(filepath)

    image_url = f"/static/uploads/{filename}"

    # Real time
    cooked_time = datetime.now().strftime("%d %b %Y, %I:%M %p")

    memories = session.get("memories", [])

    memories.append({
        "name": name,
        "calories": calories,
        "ingredients": ingredients,
        "image": image_url,
        "note": note,
        "cooked_at": cooked_time
    })

    session["memories"] = memories

    return jsonify({"message": "Saved"})
@app.route("/get-memories")
def get_memories():
    if "user_id" not in session:
        return jsonify([])
    return jsonify(session.get("memories", []))

@app.route("/delete-memory", methods=["POST"])
def delete_memory():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    index = request.json.get("index")
    memories = session.get("memories", [])

    if 0 <= index < len(memories):
        memories.pop(len(memories) - 1 - index)

    session["memories"] = memories
    return jsonify({"message": "Deleted"})

@app.route("/analytics-data")
def analytics_data():
    if "memories" not in session:
        return jsonify({
            "totalCalories": 0,
            "totalRecipes": 0,
            "topIngredient": "-",
            "weeklyCalories": [0]*7,
            "ingredientLabels": [],
            "ingredientCounts": []
        })

    memories = session["memories"]

    total_calories = 0
    ingredient_counter = Counter()
    weekly_calories = [0]*7  # Mon-Sun

    for mem in memories:
        # Calories
        try:
            cal = int(mem.get("calories", 0))
        except:
            cal = 0

        total_calories += cal

        # Weekly grouping
        cooked_time = mem.get("cooked_at")
        if cooked_time:
            dt = datetime.strptime(cooked_time, "%d %b %Y, %I:%M %p")
            weekday = dt.weekday()  # Monday=0
            weekly_calories[weekday] += cal

        # Ingredients count
        ingredients = mem.get("ingredients", [])
        for ing in ingredients:
            ingredient_counter[ing] += 1

    top_ingredient = ingredient_counter.most_common(1)
    top_ingredient = top_ingredient[0][0] if top_ingredient else "-"

    labels = list(ingredient_counter.keys())
    counts = list(ingredient_counter.values())

    return jsonify({
        "totalCalories": total_calories,
        "totalRecipes": len(memories),
        "topIngredient": top_ingredient,
        "weeklyCalories": weekly_calories,
        "ingredientLabels": labels,
        "ingredientCounts": counts
    })


# ================= RUN =================
import os

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

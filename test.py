from flask import Flask, session
from flask_cors import CORS
from flask_session import Session

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "http://localhost:5000"}})
app.config['SECRET_KEY'] =  "asdasdasdasda"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

@app.get("/set_foo")
def set_foo():
    session["foo"] = "bar"
    return session["foo"]

@app.get("/get_foo")
def get_foo():
    return session["foo"]

if __name__=="__main__":
    app.run("localhost", 5000)
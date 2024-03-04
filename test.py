from flask import Flask, session
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, 
     supports_credentials=True, 
     origin='http://127.0.0.1:5500')
app.config["SESSION_COOKIE_SECURE"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "None"

app.config['SECRET_KEY'] =  "asdasdasdasda"

@app.post("/set_foo")
def set_foo():
    session["foo"] = "bar"
    return session.get("foo", "undefined")

@app.get("/get_foo")
def get_foo():
    return session.get("foo", "undefined")

if __name__=="__main__":
    app.run("localhost", 5000, debug=True)
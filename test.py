from flask import Flask, session
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.secret_key = "123"

@app.get("/set_foo")
def set_foo():
    session["foo"] = "bar"
    return session["foo"]

@app.get("/get_foo")
def get_foo():
    return session["foo"]

if __name__=="__main__":
    app.run("localhost", 5000, debug=True)
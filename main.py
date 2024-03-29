from flask import Flask, jsonify
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

@app.route('/')
def hello():
    iris = load_iris()
    X, y = iris.data, iris.target
    model = RandomForestClassifier()
    model.fit(X, y)
    return jsonify({'message': 'Model trained successfully!'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
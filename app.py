from flask import Flask, render_template, request
import random

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Input values
        tenure = float(request.form.get("tenure", 0))
        monthly = float(request.form.get("monthly", 0))
        total = float(request.form.get("total", 0))

        # Demo logic for probability
        prob = int(
            min(
                100,
                max(
                    0,
                    (monthly / 10) + (50 - tenure) + random.randint(-10, 10)
                )
            )
        )

        prediction = "Churn" if prob >= 50 else "No Churn"

        # Optional demo rule
        if tenure <= 3 and monthly >= 80:
            prediction = "Churn"
            prob = max(prob, 75)

        return render_template("result.html", prediction=prediction, prob=prob)

    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)

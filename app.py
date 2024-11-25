from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained machine learning model
model = pickle.load(open('logistic_model.pkl', 'rb'))

@app.route("/", methods=["GET", "POST"])
def predict():
    diagnosis = None  # Default value for the diagnosis
    
    if request.method == "POST":
        try:
            # Extract input values from the form
            id_value = request.form.get("id")  # Get the ID (currently unused in prediction)
            input_features = [
                float(request.form.get("radius_mean")),
                float(request.form.get("texture_mean")),
                float(request.form.get("perimeter_mean")),
                float(request.form.get("area_mean")),
                float(request.form.get("smoothness_mean")),
                float(request.form.get("compactness_mean")),
                float(request.form.get("concavity_mean")),
                float(request.form.get("concave_points_mean")),
                float(request.form.get("symmetry_mean")),
                float(request.form.get("fractal_dimension_mean")),
                float(request.form.get("radius_se")),
                float(request.form.get("texture_se")),
                float(request.form.get("perimeter_se")),
                float(request.form.get("area_se")),
                float(request.form.get("smoothness_se")),
                float(request.form.get("compactness_se")),
                float(request.form.get("concavity_se")),
                float(request.form.get("concave_points_se")),
                float(request.form.get("symmetry_se")),
                float(request.form.get("fractal_dimension_se")),
                float(request.form.get("radius_worst")),
                float(request.form.get("texture_worst")),
                float(request.form.get("perimeter_worst")),
                float(request.form.get("area_worst")),
                float(request.form.get("smoothness_worst")),
                float(request.form.get("compactness_worst")),
                float(request.form.get("concavity_worst")),
                float(request.form.get("concave_points_worst")),
                float(request.form.get("symmetry_worst")),
                float(request.form.get("fractal_dimension_worst")),
            ]

            # Log inputs (optional, for debugging purposes)
            print(f"ID: {id_value}, Inputs: {input_features}")
            print(f"Inputs shape: {len(input_features)}")

            # Convert inputs into a NumPy array and reshape for prediction
            input_array = np.array(input_features).reshape(1, -1)

            # Predict using the trained model
            prediction = model.predict(input_array)

            # Map prediction to human-readable label
            diagnosis = "Malignant" if prediction[0] == 1 else "Benign"

        except ValueError as e:
            print(f"ValueError occurred: {e}")
            diagnosis = "Breast Cancer Predicted"

        except Exception as e:
            print(f"Error occurred: {e}")
            diagnosis = "Breast Cancer Predicted"

    # Render the HTML template with the prediction result
    return render_template("index.html", diagnosis=diagnosis)

if __name__ == "__main__":
    app.run(debug=True)

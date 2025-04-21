import streamlit as st
import pandas as pd
import joblib
import json
import zipfile
import tempfile
import os
import glob

st.set_page_config(page_title="OptiML Prediction App", layout="wide")
st.title("OptiML - Prediction App")
st.write("If you don't have a model package, please go to the [OptiML Suite](https://yogeshsj.vercel.app/) to create one.")

uploaded_zip = st.file_uploader("Upload Model Package (.zip)", type="zip")

if uploaded_zip:
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                zip_ref.extractall(tmp_dir)

            json_files = glob.glob(os.path.join(tmp_dir, "artifacts/model_inputs.json"), recursive=True)
            model_files = glob.glob(os.path.join(tmp_dir, "artifacts/best_model.pkl"), recursive=True)
            encoder_files = glob.glob(os.path.join(tmp_dir, "artifacts/label_encoders.pkl"), recursive=True)

            if not json_files:
                st.error("❌ model_inputs.json not found in the ZIP file.")
                st.stop()

            if not model_files:
                st.error("❌ best_model.pkl not found in the ZIP file.")
                st.stop()

            if not encoder_files:
                st.error("❌ label_encoders.pkl not found in the ZIP file.")
                st.stop()

            json_path = json_files[0]
            model_path = model_files[0]
            encoder_path = encoder_files[0]

            with open(json_path, "r") as f:
                model_inputs = json.load(f)

            model = joblib.load(model_path)
            label_encoders = joblib.load(encoder_path)

            st.success("✅ Model package loaded successfully!")

        except Exception as e:
            st.error(f"❌ Failed to load model package: {e}")
            st.stop()

        input_columns = model_inputs.get("input_columns", {})
        target_info = model_inputs.get("target", {})

        st.subheader("Enter Inputs for Prediction")

        user_input = {}

        for key, col_info in input_columns.items():
            name = col_info["variable_name"]
            vtype = col_info["variable_type"]

            if vtype == "Numeric":
                value = st.number_input(f"{name} (Numeric)", value=0.0)
            elif vtype in ["Binary", "Categorical"]:
                options = list(col_info["inputs"].values())
                value = st.selectbox(f"{name} ({vtype})", options)
            else:
                value = st.text_input(f"{name} (Text)", value="")
            user_input[name] = value

        if st.button("Predict"):
            try:
                input_df = pd.DataFrame([user_input])

                for col, encoder in label_encoders.items():
                    if col in input_df.columns:
                        try:
                            input_df[col] = encoder.transform(input_df[col])
                        except Exception:
                            st.error(f"❌ Invalid input for encoded column: {col}")
                            st.stop()

                prediction = model.predict(input_df)[0]
                t_name = target_info["variable_name"]
                st.success(f"{t_name.capitalize()}: **{prediction}**")

            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")

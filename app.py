import streamlit as st
import pandas as pd
from pycaret.classification import *
from rdkit import Chem
from rdkit.Chem import Descriptors
import joblib
import plotly.express as px
import os

# Streamlit app configuration
st.set_page_config(page_title="LigandScope", page_icon="🧪", layout="wide")
st.title("🧪 LigandScope: Drug Discovery Web App")
st.markdown("A code-free platform for ligand screening, model training, and predictions in computational drug discovery.")

# Ensure models directory exists
if not os.path.exists("models"):
    os.makedirs("models")

# Session state to store data and models
if 'df' not in st.session_state:
    st.session_state.df = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'descriptors_df' not in st.session_state:
    st.session_state.descriptors_df = None

# Function to generate molecular descriptors using RDKit
def generate_descriptors(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list if s]
    descriptors = []
    for mol in mols:
        if mol:
            desc = {
                'MolecularWeight': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'NumHDonors': Descriptors.NumHDonors(mol),
                'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                'NumRotatableBonds': Descriptors.NumRotatableBonds(mol)
            }
            descriptors.append(desc)
        else:
            descriptors.append({
                'MolecularWeight': None,
                'LogP': None,
                'TPSA': None,
                'NumHDonors': None,
                'NumHAcceptors': None,
                'NumRotatableBonds': None
            })
    return pd.DataFrame(descriptors)

# Sidebar navigation
st.sidebar.title("Navigation")
menu = ["Upload & Preprocess Data", "Train Model", "Predict with Model"]
choice = st.sidebar.selectbox("Select Option", menu)

# Upload & Preprocess Data
if choice == "Upload & Preprocess Data":
    st.header("Step 1: Upload & Preprocess Ligand Dataset")
    st.markdown("Upload a CSV file containing SMILES strings and target labels.")
    data_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if data_file:
        df = pd.read_csv(data_file)
        st.session_state.df = df
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        # Select SMILES and target columns
        st.subheader("Configure Dataset")
        smiles_col = st.selectbox("Select SMILES Column", df.columns, key="smiles_col")
        target_col = st.selectbox("Select Target Column", df.columns, key="target_col")
        
        if st.button("Generate Molecular Descriptors"):
            with st.spinner("Generating descriptors..."):
                descriptors_df = generate_descriptors(df[smiles_col])
                st.session_state.descriptors_df = descriptors_df
                st.session_state.df = pd.concat([df, descriptors_df], axis=1)
                st.subheader("Dataset with Descriptors")
                st.dataframe(st.session_state.df.head())
                st.download_button(
                    label="Download Processed Dataset",
                    data=st.session_state.df.to_csv(index=False),
                    file_name="processed_ligand_data.csv",
                    mime="text/csv"
                )

# Train Model
elif choice == "Train Model":
    st.header("Step 2: Train Machine Learning Model")
    if st.session_state.df is None:
        st.error("Please upload and preprocess data first.")
    else:
        st.markdown("Train a model using PyCaret's automated machine learning pipeline.")
        target_col = st.selectbox("Select Target Column", st.session_state.df.columns, key="train_target_col")
        model_types = st.multiselect(
            "Select Models to Compare",
            ['rf', 'xgboost', 'lightgbm', 'lr', 'svm'],
            default=['rf', 'xgboost', 'lightgbm']
        )
        
        if st.button("Train Model"):
            with st.spinner("Setting up and training models..."):
                try:
                    clf = setup(
                        data=st.session_state.df,
                        target=target_col,
                        silent=True,
                        html=False,
                        use_gpu=False
                    )
                    st.session_state.best_model = compare_models(include=model_types, sort='AUC')
                    save_model(st.session_state.best_model, 'models/ligand_model')
                    st.success("Model trained and saved as 'ligand_model.pkl'!")
                    
                    # Display model comparison
                    st.subheader("Model Comparison")
                    comparison = pull()
                    st.dataframe(comparison)
                    
                    # Plot ROC curve
                    st.subheader("ROC Curve")
                    st.pyplot(plot_model(st.session_state.best_model, plot='auc', save=True))
                    
                    # Plot feature importance
                    st.subheader("Feature Importance")
                    st.pyplot(plot_model(st.session_state.best_model, plot='feature', save=True))
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")

# Predict with Model
elif choice == "Predict with Model":
    st.header("Step 3: Make Predictions")
    st.markdown("Use a trained model or upload a pre-trained model to predict ligand activity.")
    model_choice = st.radio("Model Source", ["Use Trained Model", "Upload Pre-trained Model"])
    
    model = None
    if model_choice == "Use Trained Model":
        if st.session_state.best_model is None:
            st.error("No trained model available. Please train a model first.")
        else:
            model = st.session_state.best_model
    else:
        model_file = st.file_uploader("Upload Pre-trained Model (.pkl)", type=["pkl"])
        if model_file:
            with open("models/uploaded_model.pkl", "wb") as f:
                f.write(model_file.read())
            model = load_model("models/uploaded_model")
    
    if model:
        st.subheader("Upload Data for Prediction")
        pred_file = st.file_uploader("Upload CSV file", type=["csv"], key="pred_file")
        if pred_file:
            pred_df = pd.read_csv(pred_file)
            smiles_col = st.selectbox("Select SMILES Column", pred_df.columns, key="pred_smiles_col")
            
            if st.button("Generate Predictions"):
                with st.spinner("Generating descriptors and predictions..."):
                    pred_descriptors = generate_descriptors(pred_df[smiles_col])
                    pred_df_full = pd.concat([pred_df, pred_descriptors], axis=1)
                    predictions = predict_model(model, data=pred_df_full)
                    
                    # Display predictions
                    st.subheader("Prediction Results")
                    st.dataframe(predictions)
                    st.download_button(
                        label="Download Predictions",
                        data=predictions.to_csv(index=False),
                        file_name="ligand_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Plot prediction distribution
                    st.subheader("Prediction Distribution")
                    fig = px.histogram(predictions, x="Label", title="Distribution of Predicted Classes")
                    st.plotly_chart(fig)
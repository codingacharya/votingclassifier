import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Streamlit App Title
st.title("🧠 Ensemble Learning - Voting Classifier")

# File upload section
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.write("📊 Preview of the uploaded dataset:")
    st.dataframe(df.head())

    # Feature and target selection
    st.sidebar.header("🔢 Feature Selection")
    target_column = st.sidebar.selectbox("Select Target Variable", df.columns)
    feature_columns = st.sidebar.multiselect("Select Feature Columns", df.columns, default=[col for col in df.columns if col != target_column])

    if target_column and feature_columns:
        # Prepare data
        X = df[feature_columns]
        y = df[target_column]

        # Encode categorical target variable (if necessary)
        if y.dtype == 'O':  
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardization (for Logistic Regression & KNN)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Sidebar - Model Selection
        st.sidebar.header("🤖 Choose Models")
        use_logistic = st.sidebar.checkbox("Use Logistic Regression", value=True)
        use_knn = st.sidebar.checkbox("Use K-Nearest Neighbors", value=True)
        use_dt = st.sidebar.checkbox("Use Decision Tree", value=True)

        # Voting Type
        voting_type = st.sidebar.radio("⚖️ Voting Type", ("Hard Voting", "Soft Voting"))

        # Define models
        models = []
        if use_logistic:
            models.append(("Logistic Regression", LogisticRegression()))
        if use_knn:
            models.append(("KNN", KNeighborsClassifier(n_neighbors=5)))
        if use_dt:
            models.append(("Decision Tree", DecisionTreeClassifier(max_depth=3)))

        # Ensure at least one model is selected
        if not models:
            st.warning("⚠️ Please select at least one model to proceed!")
        else:
            # Create Voting Classifier
            voting_clf = VotingClassifier(estimators=models, voting='hard' if voting_type == "Hard Voting" else 'soft')

            # Train model
            voting_clf.fit(X_train, y_train)

            # Make predictions
            y_pred = voting_clf.predict(X_test)

            # Compute accuracy
            accuracy = accuracy_score(y_test, y_pred)
            st.success(f"✅ **{voting_type} Accuracy:** {accuracy:.4f}")

else:
    st.info("📥 Please upload a CSV dataset to begin.")


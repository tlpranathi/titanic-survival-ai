import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# PRE PROCESSING FUNCTION
def pre_process(df):
    df = df.copy()

    # filling missing age with median per pclass
    if "Age"  and "Pclass" in df.columns:
        df["Age"] = df.groupby("Pclass")["Age"].transform(lambda x: x.fillna(x.median()))

    # drop cabin column
    if "Cabin" in df.columns:
        df.drop("Cabin", axis=1, inplace=True)

    # drop passenger id column
    if "PassengerId" in df.columns:
        df.drop("PassengerId", axis=1, inplace=True)

    # handle missing fare (if any in test set)
    if "Fare" in df.columns:
        df["Fare"].fillna(df["Fare"].median(), inplace=True)

    # handle missing esmbarked
    if "Embarked" in df.columns:
        df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
        
    # One-hot encode categorical features
    if "Sex" in df.columns:
        sex = pd.get_dummies(df["Sex"], drop_first=True)
        df = pd.concat([df.drop("Sex", axis=1), sex], axis=1)

    if "Embarked" in df.columns:
        embark = pd.get_dummies(df["Embarked"], drop_first=True)
        df = pd.concat([df.drop("Embarked", axis=1), embark], axis=1)

    # drop text columns
    for col in ["Name", "Ticket"]:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    return df


# MODEL EVALUATION FUNCTION
# running cross validation with multiple metrics
def evaluate_models(X, y, models, cv=5):
    results = {}
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=metrics)
        model_scores = {metric: np.mean(scores[f'test_{metric}']) for metric in metrics} # mean of each metric (dictionary comprehension)
        results[name] = model_scores

        # print them
        print(f"\n{name}:")
        for metric, score in model_scores.items():
            print(f"{metric}: {score:.3f}")
    
    return results

# feature importance for random tree model
def feat_imp(rfmodel, X_train):
    importances = rfmodel.feature_importances_
    df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances}).sort_values(by='Importance', ascending=True)
    
    st.subheader('Random Forest Feature Importance')
    fig, ax = plt.subplots(figsize=(8, 6))
    # dark background for the plot
    dark = '#0e1117'            
    grid = '#31343f'           

    fig.patch.set_facecolor(dark)
    ax.set_facecolor(dark)

    sns.barplot(x='Importance', y='Feature', data=df, palette='viridis', ax=ax)
    ax.grid(True, axis='x', linestyle='--', linewidth=0.7, color=grid, alpha=0.6)
    ax.set_xlabel('Importance', color='white')
    ax.set_ylabel('Feature', color='white')
    ax.set_title('Random Forest Feature Importance', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color(grid)

    ax.invert_yaxis()
    st.pyplot(fig)



# STREAMLIT APP
st.title("Titanic Survival AI 🚢🛟")

st.write("Upload a **training CSV** (must include the 'Survived' column) to build the model, and either a **test CSV** (without 'Survived') or enter details for a single passenger to make predictions")

train_file = st.file_uploader("Upload Train CSV", type="csv")
test_file = st.file_uploader("Upload Test CSV", type="csv")

# proceed if training file is uploaded
if train_file is not None:
    train = pd.read_csv(train_file)
    st.write("### Training Data Preview")
    st.dataframe(train.head())

    if "Survived" not in train.columns:
        st.warning("training file must contain 'Survived' column")
    else:
        train_clean = pre_process(train) # pre process training data
        X_train = train_clean.drop("Survived", axis=1)
        y_train = train_clean["Survived"]

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000), "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5), "SVM": SVC(kernel="rbf", C=1, gamma="scale")
        }

        # run cross validation
        if st.button("Run Cross-Validation"):
            with st.spinner("Running cross-validation..."):
                results = evaluate_models(X_train, y_train, models, cv=5)
                st.write("### Cross-Validation Results")
                st.dataframe(pd.DataFrame(results).T.style.format("{:.3f}"))

        # test set prediction
        if test_file is not None:
            test = pd.read_csv(test_file) 
            X_test = pre_process(test) # pre process test data

            if st.button("Train on Train and Predict Test"):
                if "PassengerId" in test.columns:
                    test_ids = test["PassengerId"]
                else:
                    test_ids = range(len(X_test))

                with st.spinner("Training models and predicting test set..."):
                    predictions = {}
                    for name, model in models.items():
                        model.fit(X_train, y_train)
                        predictions[name] = model.predict(X_test)

                    preds_df = pd.DataFrame({"PassengerId": test_ids})
                    for name, prediction in predictions.items():
                        preds_df[name] = prediction
                        
                    st.write("### Test Predictions (first 10 rows)")
                    st.dataframe(preds_df.head(10))
                        
                    st.download_button(label="Download Predictions CSV", data=preds_df.to_csv(index=False).encode("utf-8"), 
                                        file_name="titanic_predictions.csv", mime="text/csv")
            
    # user input prediction
    st.write("### Predict Survival for a Single Passenger")
    with st.form("single_passenger_form"):
        col1, col2 = st.columns(2)
        with col1:
            pclass = st.selectbox("Passenger Class", (1, 2, 3))
            sex = st.selectbox("Sex", ("male", "female"))
            sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=8, value=0)
        with col2:
            age = st.number_input("Age", min_value=0.0, max_value=200.0, value=25.0)
            parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=6, value=0)
            fare = st.number_input("Fare", min_value=0.0, value=40.0)

        submitted = st.form_submit_button("Predict Survival")

        if submitted:
            # data frame for the new input
            input_df = pd.DataFrame([{"Pclass": pclass, "Age": age, "SibSp": sibsp, "Parch": parch, "Fare": fare,
                                    "male": 1 if sex == "male" else 0, "C": 0, "Q": 0, "S": 1}])

            # align the columns
            input_df = input_df.reindex(columns=X_train.columns, fill_value=0)
                
            # prediction using random forest
            model_name = "Random Forest"
            model = models[model_name]
            model.fit(X_train, y_train)
                
            prediction = model.predict(input_df)[0]
            prediction_prob = model.predict_proba(input_df)[0] #  returns [prob_death, prob_survive]
                
            st.subheader("Prediction Result")
            if prediction == 1:
                st.success(f"Based on the data, the model predicts this person **would have survived**")
            else:
                st.error(f"Based on the data, the model predicts this person **would NOT have survived**")
                
            st.write(f"Model: {model_name}")
            st.write(f"Survival Probability: **{prediction_prob[1]:.3%}**")
            st.write(f"Death Probability: **{prediction_prob[0]:.3%}**")
            
            rfmodel = RandomForestClassifier(n_estimators=100, random_state=42)
            rfmodel.fit(X_train, y_train)

            # feature importance
            feat_imp(rfmodel, X_train)  
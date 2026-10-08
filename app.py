import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay

st.set_page_config(page_title="Student Performance model")

st.title("Student Performance model")
st.write("Predict if a student performance is High, Average or Low")


df=pd.read_csv("xAPI-Edu-Data.csv")

X=df.drop("Class",axis=1)
y=df["Class"]

categorical_cols=X.select_dtypes(include=["object"]).columns
numerical_cols=X.select_dtypes(exclude=["object"]).columns

preprocessor=ColumnTransformer([
    ("cat",OneHotEncoder(handle_unknown="ignore"),categorical_cols),
    ("num","passthrough",numerical_cols)
])

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

X_train_encoded=preprocessor.fit_transform(X_train)
X_test_encoded=preprocessor.transform(X_test)

model=RandomForestClassifier(n_estimators=200,random_state=42)
model.fit(X_train_encoded,y_train)

y_pred=model.predict(X_test_encoded)
accuracy=accuracy_score(y_test,y_pred)

st.subheader("Model Performance")
st.write("Accuracy:",round(accuracy*100,2),"%")
st.write("Total Students:",len(df))

st.subheader("Enter Student Details")

col1,col2,col3=st.columns(3)

with col1:
    gender=st.selectbox("Gender",sorted(df["gender"].unique()))
    nationality=st.selectbox("Nationality",sorted(df["NationalITy"].unique()))
    birthplace=st.selectbox("Place of Birth",sorted(df["PlaceofBirth"].unique()))
    stage=st.selectbox("Stage",sorted(df["StageID"].unique()))
    grade=st.selectbox("Grade",sorted(df["GradeID"].unique()))

with col2:
    section=st.selectbox("Section",sorted(df["SectionID"].unique()))
    topic=st.selectbox("Topic",sorted(df["Topic"].unique()))
    semester=st.selectbox("Semester",sorted(df["Semester"].unique()))
    relation=st.selectbox("Parent Relation",sorted(df["Relation"].unique()))
    parent_survey=st.selectbox(
        "Parent Answering Survey",
        sorted(df["ParentAnsweringSurvey"].unique())
    )

with col3:
    parent_satisfaction=st.selectbox(
        "Parent School Satisfaction",
        sorted(df["ParentschoolSatisfaction"].unique())
    )

    absence=st.selectbox(
        "Student Absence Days",
        sorted(df["StudentAbsenceDays"].unique())
    )

    raised_hands=st.slider("Raised Hands",0,100,50)
    visited_resources=st.slider("Visited Resources",0,100,50)
    announcements=st.slider("Announcements Viewed",0,100,30)
    discussion=st.slider("Discussion",0,100,30)

student=pd.DataFrame([{
    "gender":gender,
    "NationalITy":nationality,
    "PlaceofBirth":birthplace,
    "StageID":stage,
    "GradeID":grade,
    "SectionID":section,
    "Topic":topic,
    "Semester":semester,
    "Relation":relation,
    "raisedhands":raised_hands,
    "VisITedResources":visited_resources,
    "AnnouncementsView":announcements,
    "Discussion":discussion,
    "ParentAnsweringSurvey":parent_survey,
    "ParentschoolSatisfaction":parent_satisfaction,
    "StudentAbsenceDays":absence
}])

if st.button("Predict Student Performance"):

    student_encoded=preprocessor.transform(student)

    prediction=model.predict(student_encoded)[0]
    probability=model.predict_proba(student_encoded)[0]

    confidence=probability.max()*100

    labels={
        "H":"High",
        "M":"Average",
        "L":"Low"
    }

    result=labels[prediction]

    st.subheader("Prediction")

    if result=="High":
        st.success("Performance: "+result)
    elif result=="Average":
        st.warning("Performance: "+result)
    else:
        st.error("Performance: "+result)

    st.write("Confidence:",round(confidence,2),"%")

st.subheader("Important Factors")

feature_names=preprocessor.get_feature_names_out()

importance_df=pd.DataFrame({
    "Feature":feature_names,
    "Importance":model.feature_importances_
})

importance_df=importance_df.sort_values(
    "Importance",ascending=False
).head(10)

st.bar_chart(importance_df.set_index("Feature"))

st.subheader("Confusion Matrix")

cm=confusion_matrix(y_test,y_pred,labels=model.classes_)

fig,ax=plt.subplots()

disp=ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot(ax=ax)

st.pyplot(fig)


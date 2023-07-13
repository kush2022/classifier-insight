import streamlit as st 
from pathlib import Path
# from streamlit_lottie import st_lottie
import json

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris

import matplotlib.pyplot as plt 
import pandas as pd



#--------- PATH CONFIG -----
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
machine_lottie = current_dir / 'lottie' / 'machine-learning.json'
sidelottie  = current_dir / 'lottie' / 'data.json'
css_file = current_dir/ 'styles'/'main.css'



# ---------- PAGE CONFIG --------------
st.set_page_config(
    page_title="Empowering Insights",
    page_icon="🧊",
    layout="wide",
)



# --- LOAD CSS --------
@st.cache_data
def load_css(path):
    with open(css_file) as f:
        css = f.read()
        return f'<style>{css}</style>'
css = load_css(css_file)
st.markdown(css, unsafe_allow_html=True)


#------ HEADER ----------------
col1, col2 = st.columns(2, gap='large')


with col1:
    st.title('Explore Classifiers: Unleash Insights')
    st.write('Uncover the Power of Machine Learning')

with open(machine_lottie, 'r') as file:
    url = json.load(file)
# with open(sidelottie, 'r') as file:
#     sidelottie = json.load(file)



# with col2:
#     st_lottie(url, height=120, width=120)


# with st.sidebar:
#     st_lottie(sidelottie, height=130, width=130)




#------ CHOOSE THE DATASET ----------
dataset_name = st.sidebar.selectbox("Select Dataset", ('Iris', 'Breast Cancer', 'Wine dataset'))



#------ CLASSIFIER -----------------
classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM","Logistic Regression", "Decision Tree", "Random Forest"))



#------ LOAD DATASET ----------------
@st.cache_data
def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target

    return X, y 

X, y = get_dataset(dataset_name)

#------ ADD PARAMETER --------------
def add_params(clf_name):
    params = dict()

    if clf_name == "KNN":
        k = st.sidebar.slider('K', 1, 15)
        params['K'] = k
    elif clf_name == "SVM":
        C = st.sidebar.slider('C', 0.01, 10.0)
        kernel = st.sidebar.selectbox('kernel', ('rbf', 'linear'))
        params['C'] = C
        params['kernel'] = kernel 
    elif clf_name == "Logistic Regression":
        params = {}
    elif clf_name == "Decision Tree":
        params = {}
    else:
        params = {}
    

    return params 

params = add_params(classifier_name)

#------ CLASSIFIER ------
def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name == "SVM":
        clf = SVC(C=params['C'], kernel=params['kernel'])
    elif clf_name == "Logistic Regression":
        clf = LogisticRegression()
    elif clf_name == "Decision Tree":
        clf = DecisionTreeClassifier()
    else:
        clf = RandomForestClassifier()
    return clf 
clf = get_classifier(classifier_name, params)


#----- TRAIN TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)
acc = accuracy_score(y_test, y_predict)


st.markdown("""
### Data Info
""")
st.write('Shape', X.shape)


#------ HERO SECTION --------------
UI = {
    'Data-set': dataset_name,
    'classifier': classifier_name,
    'Accuracy': acc
    
}

with st.container():
    st.divider()
    cols = st.columns(len(UI))
    for index, (platform, link) in enumerate(UI.items()):
        cols[index].write(f"{platform}: {link}")


#----- PCA -----------
pca = PCA(n_components=2)
X_projected = pca.fit_transform(X)
x1 = X_projected[:,0]
x2 = X_projected[:,1]

fig = plt.figure()
plt.scatter(x1, x2, alpha=0.8, cmap='viridis')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.colorbar()
st.pyplot(fig)


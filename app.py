import streamlit as st
import google.generativeai as genai
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import json
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Multilingual STEM Teaching Assistant", layout="wide")

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown('<p class="title">Multilingual STEM Teaching Assistant</p>', unsafe_allow_html=True)

# ----------------------------
# Gemini API Setup
# ----------------------------

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

model = genai.GenerativeModel("gemini-1.5-flash")

def generate_ai(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except:
        return "AI could not generate response. Try again."

# ----------------------------
# Sidebar
# ----------------------------

st.sidebar.title("Learning Controls")

difficulty = st.sidebar.selectbox(
    "Difficulty",
    ["Beginner","Intermediate","Advanced"]
)

language = st.sidebar.selectbox(
    "Language",
    ["English","Tamil","Hindi","Japanese"]
)

mode = st.sidebar.selectbox(
    "Feature",
    [
        "AI Tutor",
        "Slides Generator",
        "Homework Generator",
        "Quiz Generator",
        "Concept Diagram",
        "Student Progress",
        "Performance Prediction"
    ]
)

# ----------------------------
# Data Storage
# ----------------------------

data_file="student_data.json"

def load_data():
    try:
        with open(data_file) as f:
            return json.load(f)
    except:
        return []

def save_data(data):
    with open(data_file,"w") as f:
        json.dump(data,f)

data = load_data()

# ----------------------------
# AI Functions
# ----------------------------

def explain(concept):

    prompt=f"""
Explain the STEM concept {concept}.

Difficulty: {difficulty}
Language: {language}

Give clear explanation with examples.
"""

    return generate_ai(prompt)


def slides(topic):

    prompt=f"""
Create slide content for teaching {topic}.

Include:
• slide titles
• bullet points
"""

    return generate_ai(prompt)


def homework(topic):

    prompt=f"""
Create 5 homework questions for {topic}.
"""

    return generate_ai(prompt)


def quiz(topic):

    prompt=f"""
Create 3 quiz questions with answers for {topic}.
"""

    return generate_ai(prompt)

# ----------------------------
# Diagram Generator
# ----------------------------

def diagram(topic):

    G = nx.Graph()

    nodes = ["Definition","Process","Example","Applications"]

    G.add_node(topic)

    for n in nodes:
        G.add_edge(topic,n)

    pos = nx.spring_layout(G)

    fig,ax = plt.subplots()

    nx.draw(
        G,pos,
        with_labels=True,
        node_color="#38bdf8",
        node_size=3000
    )

    return fig

# ----------------------------
# AI Tutor
# ----------------------------

if mode=="AI Tutor":

    concept = st.text_input("Enter STEM Concept")

    if st.button("Explain") and concept:

        with st.spinner("Generating explanation..."):

            result = explain(concept)

            data.append({"topic":concept})
            data = data[-100:]
            save_data(data)

        st.write(result)

# ----------------------------
# Slides
# ----------------------------

elif mode=="Slides Generator":

    topic = st.text_input("Topic")

    if st.button("Generate Slides") and topic:

        with st.spinner("Generating slides..."):
            st.write(slides(topic))

# ----------------------------
# Homework
# ----------------------------

elif mode=="Homework Generator":

    topic = st.text_input("Topic")

    if st.button("Generate Homework") and topic:

        with st.spinner("Generating homework..."):
            st.write(homework(topic))

# ----------------------------
# Quiz
# ----------------------------

elif mode=="Quiz Generator":

    topic = st.text_input("Topic")

    if st.button("Generate Quiz") and topic:

        with st.spinner("Generating quiz..."):
            st.write(quiz(topic))

# ----------------------------
# Diagram
# ----------------------------

elif mode=="Concept Diagram":

    topic = st.text_input("Concept")

    if st.button("Draw Diagram") and topic:
        st.pyplot(diagram(topic))

# ----------------------------
# Student Progress
# ----------------------------

elif mode=="Student Progress":

    if data:

        df = pd.DataFrame(data)

        st.dataframe(df)

        chart = df["topic"].value_counts()

        st.bar_chart(chart)

    else:
        st.info("No learning data yet.")

# ----------------------------
# Performance Prediction
# ----------------------------

elif mode=="Performance Prediction":

    hours = st.number_input("Study Hours",1,10)
    topics = st.number_input("Topics Learned",1,20)

    if st.button("Predict Score"):

        X = np.array([[1,1],[2,2],[3,3],[4,4],[5,5]])
        y = np.array([50,60,70,80,90])

        model = LinearRegression()
        model.fit(X,y)

        pred = model.predict([[hours,topics]])

        st.success(f"Predicted Score: {int(pred[0])}")
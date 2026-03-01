import streamlit as st
import requests
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import json
from sklearn.linear_model import LinearRegression
import numpy as np
import time

st.set_page_config(page_title="Multilingual STEM Teaching Assistant", layout="wide")

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown('<p class="title">Multilingual STEM Teaching Assistant</p>', unsafe_allow_html=True)


# ----------------------------
# HuggingFace API
# ----------------------------

HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"


def generate_ai(prompt):

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens":150}
    }

    try:

        response = requests.post(HF_API_URL, json=payload, timeout=30)
        result = response.json()

        # Successful response
        if isinstance(result, list):
            return result[0]["generated_text"]

        # Model loading
        if isinstance(result, dict) and "estimated_time" in result:
            time.sleep(5)
            response = requests.post(HF_API_URL, json=payload)
            result = response.json()

            if isinstance(result, list):
                return result[0]["generated_text"]

        # HuggingFace error
        if isinstance(result, dict) and "error" in result:
            return fallback_ai(prompt)

        return fallback_ai(prompt)

    except:
        return fallback_ai(prompt)


# ----------------------------
# Local fallback AI
# ----------------------------

def fallback_ai(prompt):

    return f"""
Explanation generated locally.

{prompt}

This concept is important in STEM education. It involves understanding the fundamental principles, real-world applications, and examples that help students grasp the idea clearly.

Key Points:
• Definition of the concept
• How it works
• Practical examples
• Applications in science and technology
"""


# ----------------------------
# Sidebar Controls
# ----------------------------

st.sidebar.title("Learning Controls")

difficulty = st.sidebar.selectbox(
    "Difficulty",
    ["Beginner","Intermediate","Advanced"]
)

language = st.sidebar.selectbox(
    "Language",
    ["English","Tamil","Hindi","Japanese","All Languages"]
)

mode = st.sidebar.selectbox(
    "Platform Feature",
    [
        "AI Tutor",
        "Doubt Solving Chatbot",
        "Educational Slides",
        "Homework Generator",
        "Interactive Quiz",
        "Lesson Planner",
        "Whiteboard Diagram",
        "Student Progress",
        "Performance Prediction",
        "Teacher Dashboard"
    ]
)


# ----------------------------
# Data Storage
# ----------------------------

data_file = "student_data.json"

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

    prompt = f"""
Explain the STEM concept '{concept}'.

Difficulty: {difficulty}

Language: {language}

Give clear explanation with examples.
"""

    return generate_ai(prompt)


def homework(topic):

    prompt=f"""
Create homework questions for {topic}.
Include 5 questions.
"""

    return generate_ai(prompt)


def slides(topic):

    prompt=f"""
Create slide content for teaching {topic}.
Include slide title and bullet points.
"""

    return generate_ai(prompt)


def quiz(topic):

    prompt=f"""
Create 3 quiz questions about {topic} with answers.
"""

    return generate_ai(prompt)


def lesson(topic):

    prompt=f"""
Create a lesson plan for teaching {topic}.
Include introduction, concepts, activities and assessment.
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

    fig, ax = plt.subplots()

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="#38bdf8",
        node_size=3000,
        font_size=10
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
# Doubt Chatbot
# ----------------------------

elif mode=="Doubt Solving Chatbot":

    question = st.text_input("Ask your doubt")

    if st.button("Solve Doubt") and question:

        with st.spinner("Thinking..."):
            answer = explain(question)

        st.write(answer)


# ----------------------------
# Slides
# ----------------------------

elif mode=="Educational Slides":

    topic = st.text_input("Enter Topic")

    if st.button("Generate Slides") and topic:

        with st.spinner("Generating slides..."):
            result = slides(topic)

        st.write(result)


# ----------------------------
# Homework
# ----------------------------

elif mode=="Homework Generator":

    topic = st.text_input("Homework Topic")

    if st.button("Generate Homework") and topic:

        with st.spinner("Generating homework..."):
            result = homework(topic)

        st.write(result)


# ----------------------------
# Quiz
# ----------------------------

elif mode=="Interactive Quiz":

    topic = st.text_input("Quiz Topic")

    if st.button("Generate Quiz") and topic:

        with st.spinner("Generating quiz..."):
            result = quiz(topic)

        st.write(result)


# ----------------------------
# Lesson Plan
# ----------------------------

elif mode=="Lesson Planner":

    topic = st.text_input("Lesson Topic")

    if st.button("Create Lesson Plan") and topic:

        with st.spinner("Creating lesson plan..."):
            result = lesson(topic)

        st.write(result)


# ----------------------------
# Diagram
# ----------------------------

elif mode=="Whiteboard Diagram":

    topic = st.text_input("Concept")

    if st.button("Draw Diagram") and topic:

        fig = diagram(topic)
        st.pyplot(fig)


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
        st.info("No student data yet.")


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


# ----------------------------
# Teacher Dashboard
# ----------------------------

elif mode=="Teacher Dashboard":

    if data:

        df = pd.DataFrame(data)

        st.metric("Total Topics Learned", len(df))

        st.bar_chart(df["topic"].value_counts())

    else:
        st.info("No student data available.")
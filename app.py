import streamlit as st
import requests
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

st.markdown(
    '<h1 class="title">Multilingual STEM Teaching Assistant</h1>',
    unsafe_allow_html=True
)

# ----------------------------
# GROQ Setup
# ----------------------------

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

def generate_ai(prompt):

    try:

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        )

        result = response.json()

        return result["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Error: {e}"

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
        "Lesson Planner",
        "Concept Diagram",
        "Student Progress",
        "Performance Prediction",
        "Teacher Dashboard"
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

    prompt = f"""
    Explain the STEM concept: {concept}

    Difficulty Level: {difficulty}

    IMPORTANT:
    Respond ONLY in {language}.

    Include:
    1. Definition
    2. Working Principle
    3. Real-world Examples
    4. Applications
    5. Summary
    """

    return generate_ai(prompt)


def slides(topic):

    prompt = f"""
    Create educational slide content for: {topic}

    IMPORTANT:
    Respond ONLY in {language}.

    Generate:

    Slide 1: Introduction

    Slide 2: Key Concepts

    Slide 3: Examples

    Slide 4: Applications

    Slide 5: Summary

    Use bullet points.
    """

    return generate_ai(prompt)


def homework(topic):

    prompt = f"""
    Create 5 homework questions on: {topic}

    Difficulty Level: {difficulty}

    IMPORTANT:
    Respond ONLY in {language}.

    Include a mix of:
    - Short Answer
    - Long Answer
    - Application Based Questions
    """

    return generate_ai(prompt)


def quiz(topic):

    prompt = f"""
    Create 5 quiz questions on: {topic}

    Difficulty Level: {difficulty}

    IMPORTANT:
    Respond ONLY in {language}.

    For each question provide:
    Question
    Answer
    """

    return generate_ai(prompt)


def lesson(topic):

    prompt = f"""
    Create a detailed lesson plan for: {topic}

    IMPORTANT:
    Respond ONLY in {language}.

    Include:

    1. Learning Objectives
    2. Introduction
    3. Teaching Content
    4. Classroom Activities
    5. Assessment
    6. Summary
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

    pos = nx.spring_layout(G, seed=42)

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

            data.append({
                "topic": concept,
                "difficulty": difficulty,
                "language": language
            })
            data = data[-100:]
            save_data(data)
 
        st.markdown(result)

# ----------------------------
# Slides
# ----------------------------

elif mode=="Slides Generator":

    topic = st.text_input("Topic")

    if st.button("Generate Slides") and topic:

        with st.spinner("Generating slides..."):
            st.markdown(slides(topic))
# ----------------------------
# Homework
# ----------------------------

elif mode=="Homework Generator":

    topic = st.text_input("Topic")

    if st.button("Generate Homework") and topic:

        with st.spinner("Generating homework..."):
            st.markdown(homework(topic))

# ----------------------------
# Quiz
# ----------------------------

elif mode=="Quiz Generator":

    topic = st.text_input("Topic")

    if st.button("Generate Quiz") and topic:

        with st.spinner("Generating quiz..."):
            st.markdown(quiz(topic))

# ----------------------------
# Lesson Planner
# ----------------------------

elif mode=="Lesson Planner":

    topic = st.text_input("Lesson Topic")

    if st.button("Create Lesson Plan") and topic:

        with st.spinner("Creating lesson plan..."):

            result = lesson(topic)

        st.markdown(result)

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

# ----------------------------
# Teacher Dashboard
# ----------------------------

elif mode=="Teacher Dashboard":

    if data:

        df = pd.DataFrame(data)

        st.metric(
            "Total Topics Learned",
            len(df)
        )

        st.bar_chart(
            df["topic"].value_counts()
        )

    else:
        st.info("No student data available.")
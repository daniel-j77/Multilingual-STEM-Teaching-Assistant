# Multilingual STEM Teaching Assistant

## Overview

Multilingual STEM Teaching Assistant is an AI-powered educational platform built using Streamlit and Groq API. It helps students and teachers learn STEM concepts through AI-generated explanations, quizzes, homework, lesson plans, and concept diagrams.

The application supports multiple languages including:

* English
* Tamil
* Hindi
* Japanese

---

## Features

### AI STEM Tutor

* Explains STEM concepts in simple and detailed formats.
* Supports Beginner, Intermediate, and Advanced difficulty levels.
* Generates responses in the selected language.

### Slides Generator

* Creates educational slide content.
* Generates structured slide titles and bullet points.

### Homework Generator

* Creates homework questions based on selected topics.
* Supports different difficulty levels.

### Quiz Generator

* Generates quiz questions with answers.
* Useful for self-assessment and classroom activities.

### Lesson Planner

* Creates complete lesson plans including:

  * Learning Objectives
  * Introduction
  * Teaching Content
  * Activities
  * Assessment
  * Summary

### Concept Diagram Generator

* Generates concept relationship diagrams using NetworkX and Matplotlib.

### Student Progress Tracking

* Stores recently learned topics.
* Displays topic learning history.

### Performance Prediction

* Predicts student performance using a machine learning model.

### Teacher Dashboard

* Displays analytics on learned topics.
* Provides visual insights using charts.

---

## Technologies Used

* Python
* Streamlit
* Groq API
* Requests
* Pandas
* NumPy
* Scikit-Learn
* NetworkX
* Matplotlib

---

## Project Structure

```text
Multilingual-STEM-Teaching-Assistant/
│
├── app.py
├── style.css
├── student_data.json
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/multilingual-stem-teaching-assistant.git
```

Move into the project folder:

```bash
cd multilingual-stem-teaching-assistant
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create:

```text
.streamlit/secrets.toml
```

Add your Groq API key:

```toml
GROQ_API_KEY="your_groq_api_key"
```

Run the application:

```bash
streamlit run app.py
```

---

## Deployment

This project can be deployed using Streamlit Community Cloud.

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app.
4. Select the repository.
5. Set `app.py` as the main file.
6. Add the Groq API key in Streamlit Secrets.
7. Deploy the application.

---

## Future Enhancements

* Voice-based learning assistant
* PDF export for notes and lesson plans
* Student login system
* Learning analytics dashboard
* Interactive whiteboard
* AI-generated study notes
* Real-time classroom collaboration

---

## Author

Daniel J

Bachelor of Engineering in Computer Science and Engineering

Loyola Institute of Technology, Chennai

LinkedIn: https://www.linkedin.com/in/daniel-j77

GitHub: https://github.com/daniel-j77

---

## License

This project is created for educational and academic purposes.

import streamlit as st
from crewai import Agent, Task, Process, Crew
from langchain_community.llms import Ollama
import torch

st.title("Multi-Agent Developer Chatbot")

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Initialize models
if "llm_model" not in st.session_state:
    st.session_state["llm_model"] = Ollama(model="stablelm-zephyr")

# Developer Agent Selection
developer_options = {
    "App Developer": Agent(
        role="App Developer",
        goal="Provide information and guidance on mobile app development.",
        backstory="""You are an expert in mobile app development, with extensive experience in developing iOS and Android apps using 
            various technologies such as Swift, Kotlin, Flutter, and React Native. You can guide users through the process of app development, 
            from planning to deployment.""",
        verbose=True,
        allow_delegation=False,
        llm=st.session_state["llm_model"]
    ),
    "Web Developer": Agent(
        role="Web Developer",
        goal="Provide information and guidance on web development.",
        backstory="""You are an expert in web development, skilled in front-end and back-end technologies including HTML, CSS, JavaScript, 
            React, Node.js, and Django. You can assist with anything related to building and maintaining websites.""",
        verbose=True,
        allow_delegation=False,
        llm=st.session_state["llm_model"]
    ),
    "AI/ML Developer": Agent(
        role="AI/ML Developer",
        goal="Provide information and guidance on artificial intelligence and machine learning.",
        backstory="""You are an AI/ML specialist with deep knowledge of machine learning algorithms, neural networks, data science, and AI development. 
            You can help with questions about AI model development, data processing, and using AI in real-world applications.""",
        verbose=True,
        allow_delegation=False,
        llm=st.session_state["llm_model"]
    ),
    "Blockchain Developer": Agent(
        role="Blockchain Developer",
        goal="Teach about blockchain technology, Hyperledger Fabric 2.5, and Web3.",
        backstory="""You are an expert Blockchain Developer with extensive knowledge of blockchain technology, Web3, and Hyperledger Fabric 2.5. 
            You are passionate about educating others on these topics and excel at breaking down complex concepts into understandable lessons. 
            Your goal is to help others grasp both the fundamentals and advanced topics in these areas.""",
        verbose=True,
        allow_delegation=False,
        llm=st.session_state["llm_model"]
    ),
}

# User selects which developer agent to chat with
developer_choice = st.selectbox("Choose a Developer:", list(developer_options.keys()))
selected_developer = developer_options[developer_choice]

# Function to process model response
def model_res_generator(user_input):
    # Define task based on user input
    task = Task(
        description=user_input,
        expected_output="A detailed and informative response",
        agent=selected_developer,
    )

    # Create the crew with the selected developer agent and the task
    crew = Crew(
        agents=[selected_developer],
        tasks=[task],
        verbose=2,
        process=Process.sequential,
    )

    # Process user input and get the result
    result = crew.kickoff()
    
    # Return the result to be appended to chat history
    return result

# Display chat messages from history on app rerun
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input for the chatbot
if prompt := st.chat_input("Enter prompt here.."):
    # Add user message to history
    st.session_state["messages"].append({"role": "user", "content": prompt})
    
    # Display the user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get model response
    with st.chat_message("assistant"):
        response = model_res_generator(prompt)
        st.markdown(response)
        # Add assistant response to history
        st.session_state["messages"].append({"role": "assistant", "content": response})

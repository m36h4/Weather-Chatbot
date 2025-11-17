import os
import requests
import logging
from typing import List, Optional
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# --- CHANGE HERE ---
from langchain_openai import ChatOpenAI
# --------------------

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, START, END

load_dotenv()

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["OPENWEATHER_API_KEY"] = st.secrets["OPENWEATHER_API_KEY"]
# --------------------- Logging Setup ---------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("WeatherBot")


# --------------------- Weather API ---------------------
def get_weather(city: str) -> str:
    api_key = os.environ["OPENWEATHER_API_KEY"] 
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    res = requests.get(url, params=params).json()
    if res.get("cod") != 200:
        return f"Could not fetch data for {city.title()}."
    desc = res["weather"][0]["description"]
    temp = res["main"]["temp"]
    feels = res["main"]["feels_like"]
    humidity = res["main"]["humidity"]
    wind = res["wind"]["speed"]
    pressure = res["main"]["pressure"]
    return (
        f"{city.title()} — {desc}\n"
        f"{temp}°C (feels like {feels}°C)\n"
        f"Humidity: {humidity}% | Wind: {wind} m/s | Pressure: {pressure} hPa"
    )


# --------------------- LLM (OpenAI) ---------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",            # small & fast model suitable for weather bot
    temperature=0.25,
    api_key=os.environ["OPENAI_API_KEY"]
)


# --------------------- Pydantic Parser ---------------------
class Intent(BaseModel):
    intent: str = Field(description="Either 'FETCH' or 'NOT_WEATHER'.")
    cities: Optional[List[str]] = Field(default_factory=list)


parser = PydanticOutputParser(pydantic_object=Intent)


# --------------------- Prompts ---------------------
intent_prompt = ChatPromptTemplate.from_template(
    """You are a focused weather assistant.
Conversation so far:
{history}

User just said:
{user_input}

Respond ONLY as a JSON object that matches:
{format_instructions}

Rules:
- If message is about weather and contains city names → intent: 'FETCH'
- Otherwise → intent: 'NOT_WEATHER'
"""
)

weather_prompt = ChatPromptTemplate.from_template(
    """You are a weather assistant. Based on this data:
{weather_data}
Answer naturally and conversationally."""
)

intent_chain = intent_prompt | llm | parser
weather_chain = weather_prompt | llm


# --------------------- Graph State ---------------------
class AgentState(BaseModel):
    messages: List[str]
    intent: Optional[dict] = None


# --------------------- Node Functions ---------------------
def detect_intent_node(state: AgentState) -> AgentState:
    logger.info("Entering detect_intent_node")
    history_text = "\n".join(state.messages[-6:])
    start_time = datetime.now()

    try:
        result = intent_chain.invoke({
            "history": history_text,
            "user_input": state.messages[-1],
            "format_instructions": parser.get_format_instructions()
        })
        state.intent = result.dict()
    except Exception as e:
        logger.exception(f"Intent parsing failed: {e}")
        state.intent = {"intent": "NOT_WEATHER", "cities": []}

    logger.info(f"detect_intent_node completed in {(datetime.now() - start_time).total_seconds():.2f}s")
    return state


def handle_fetch_node(state: AgentState) -> AgentState:
    logger.info("Entering handle_fetch_node")
    cities = state.intent.get("cities", [])
    start_time = datetime.now()

    if not cities:
        response = "Please specify a city name."
    else:
        weather_data = "\n\n".join(get_weather(c) for c in cities)
        response = weather_chain.invoke({"weather_data": weather_data}).content

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    logger.info(f"handle_fetch_node completed in {(datetime.now() - start_time).total_seconds():.2f}s")
    return state


def handle_not_weather_node(state: AgentState) -> AgentState:
    response = "I'm designed to answer weather questions. Please ask about a city or temperature."
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    return state


# --------------------- Build Graph ---------------------
graph = StateGraph(AgentState)
graph.add_node("detect_intent", detect_intent_node)
graph.add_node("handle_fetch", handle_fetch_node)
graph.add_node("handle_not_weather", handle_not_weather_node)

graph.add_edge(START, "detect_intent")

graph.add_conditional_edges(
    "detect_intent",
    lambda s: s.intent.get("intent", "NOT_WEATHER"),
    {
        "FETCH": "handle_fetch",
        "NOT_WEATHER": "handle_not_weather"
    }
)

graph.add_edge("handle_fetch", END)
graph.add_edge("handle_not_weather", END)

app = graph.compile()


# --------------------- Streamlit UI ---------------------
st.set_page_config(page_title="Weather Chatbot", layout="centered")
st.title("Weather Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "state" not in st.session_state:
    st.session_state.state = AgentState(messages=[])

for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input("Ask about the weather..."):
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    state = st.session_state.state
    state.messages.append(user_input)

    result = app.invoke(state)
    st.session_state.state = result

    st.rerun()

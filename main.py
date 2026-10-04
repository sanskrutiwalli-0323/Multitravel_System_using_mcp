
# LangGraph Multi-Agent Travel Booking System
# with PostgreSQL Long-Term Memory

import os
import asyncio
import uuid
import operator

from typing import TypedDict, Annotated

import psycopg
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq

from mcp_client import (
    tavily_mcp_search,
    aviation_mcp_call,
    extract_destination,
    forecast_mcp_search,
    weather_mcp_search,
)


# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------

load_dotenv(override=True)

DATABASE_URL = os.getenv("DATABASE_URL")


# --------------------------------------------------
# LLM Configuration
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)


# --------------------------------------------------
# Travel State
# --------------------------------------------------

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

    user_query: str
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str
    llm_calls: int


# --------------------------------------------------
# Flight Agent Prompt
# --------------------------------------------------

FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:

1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Important:
- Do not invent confirmed flight schedules or prices.
- Clearly label estimates.
- If information is unavailable, say so.

Return concise travel guidance.
"""


# --------------------------------------------------
# Flight Agent
# --------------------------------------------------

def flight_agent(state: TravelState):

    print("\nINSIDE FLIGHT AGENT\n")

    query = state["user_query"]

    llm_calls = state.get("llm_calls", 0)

    try:

        airports = asyncio.run(
            aviation_mcp_call("list_airports")
        )

        airlines = asyncio.run(
            aviation_mcp_call("list_airlines")
        )

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:3000],
            airline_data=str(airlines)[:3000],
        )

        response = llm.invoke(
            [
                SystemMessage(
                    content="You are an expert travel flight planner."
                ),
                HumanMessage(content=prompt),
            ]
        )

        flight_data = response.content

        llm_calls += 1

    except Exception as e:

        flight_data = (
            f"Flight information unavailable: {str(e)}"
        )

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(
                content="Flight recommendations generated."
            )
        ],
        "llm_calls": llm_calls,
    }


# --------------------------------------------------
# Hotel Agent
# --------------------------------------------------

def hotel_agent(state: TravelState):

    print("\nINSIDE HOTEL AGENT\n")

    query = f"Best hotels for {state['user_query']}"

    try:

        hotel_results = asyncio.run(
            tavily_mcp_search(query)
        )

        hotel_results = str(hotel_results)

    except Exception as e:

        hotel_results = (
            f"Hotel information unavailable: {str(e)}"
        )

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(
                content="Hotel information fetched."
            )
        ],
    }


# --------------------------------------------------
# Weather Agent
# --------------------------------------------------

def weather_agent(state: TravelState):

    print("\nINSIDE WEATHER AGENT\n")

    try:

        city = extract_destination(
            state["user_query"]
        )

        weather_data = asyncio.run(
            weather_mcp_search(city)
        )

        forecast_data = asyncio.run(
            forecast_mcp_search(city)
        )

        weather_results = f"""
Destination: {city}

Current Weather:
{weather_data}

Forecast:
{forecast_data}
"""

    except Exception as e:

        weather_results = (
            f"Weather information unavailable: {str(e)}"
        )

    return {
        "weather_results": weather_results,
        "messages": [
            AIMessage(
                content="Weather information fetched."
            )
        ],
    }


# --------------------------------------------------
# Itinerary Agent
# --------------------------------------------------

def itinerary_agent(state: TravelState):

    print("\nINSIDE ITINERARY AGENT\n")

    prompt = f"""
Create a practical travel itinerary based on the
following information.

User Request:
{state.get('user_query', 'No travel request provided')}

Flight Information:
{state.get('flight_results', 'Not available')}

Hotel Information:
{state.get('hotel_results', 'Not available')}

Weather Information:
{state.get('weather_results', 'Not available')}

Instructions:
- Create a day-by-day itinerary.
- Include suitable activities and sightseeing.
- Consider the weather information.
- Use the available flight and hotel information.
- Do not invent confirmed bookings or prices.
- Clearly identify estimates and missing information.
- Mention when information needs to be verified.
"""

    try:

        response = llm.invoke(
            [
                SystemMessage(
                    content="You are an expert travel planner."
                ),
                HumanMessage(content=prompt),
            ]
        )

        itinerary = response.content

        llm_calls = state.get("llm_calls", 0) + 1

    except Exception as e:

        itinerary = (
            f"Unable to generate itinerary: {str(e)}"
        )

        llm_calls = state.get("llm_calls", 0)

    return {
        "itinerary": itinerary,
        "messages": [
            AIMessage(content=itinerary)
        ],
        "llm_calls": llm_calls,
    }


# --------------------------------------------------
# Build LangGraph
# --------------------------------------------------

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("itinerary_agent", itinerary_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "weather_agent")
graph.add_edge("weather_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", END)


# --------------------------------------------------
# PostgreSQL Checkpointer
# --------------------------------------------------

if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL is missing. "
        "Check your .env file."
    )

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
)

checkpointer = PostgresSaver(_conn)

checkpointer.setup()

app = graph.compile(
    checkpointer=checkpointer
)


# --------------------------------------------------
# Run Travel Agent
# --------------------------------------------------

if __name__ == "__main__":

    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4())
        }
    }

    user_input = input(
        "Enter travel request: "
    ).strip()

    if not user_input:
        print("Please enter a travel request.")

    else:

        initial_state = {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "itinerary": "",
            "llm_calls": 0,
        }

        try:

            result = app.invoke(
                initial_state,
                config=config,
            )

            print("\nFINAL ITINERARY:\n")
            print(result["itinerary"])

            print(
                "\nTotal LLM calls:",
                result["llm_calls"],
            )

        except Exception as e:

            print(
                "\nTravel agent failed:",
                str(e),
            )

        finally:

            _conn.close()
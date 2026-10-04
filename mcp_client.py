
import os
import asyncio

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATION_STACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


# --------------------------------------------------
# MCP Client Configuration
# --------------------------------------------------

client = MultiServerMCPClient(
    {
        # Remote Tavily MCP server
        "tavily": {
            "transport": "streamable_http",
            "url": (
                "https://mcp.tavily.com/mcp/"
                f"?tavilyApiKey={TAVILY_API_KEY}"
            ),
        },

        # Local AviationStack MCP server
        "aviationstack": {
            "transport": "stdio",
            "command": (
                r"C:\Users\sanskruti\Multi_Agent_System_With_MCP"
                r"\aviationstack-mcp\.venv\Scripts\python.exe"
            ),
            "args": [
                "-m",
                "aviationstack_mcp",
                "mcp",
                "run",
            ],
            "env": {
                "AVIATION_STACK_API_KEY": AVIATION_STACK_API_KEY,
            },
        },

        # Local custom weather MCP server
        "weather": {
            "transport": "stdio",
            "command": (
                r"C:\Users\sanskruti\Multi_Agent_System_With_MCP"
                r"\langgraph_env3\Scripts\python.exe"
            ),
            "args": [
                r"C:\Users\sanskruti\Multi_Agent_System_With_MCP"
                r"\weather_mcp_server.py"
            ],
            "env": {
                "OPENWEATHER_API_KEY": OPENWEATHER_API_KEY,
            },
        },
    }
)


# --------------------------------------------------
# Global tool variables
# --------------------------------------------------

search_tool = None

weather_tool = None
forecast_tool = None

aviation_tools = {}

_all_tools = None


# --------------------------------------------------
# Tool initialization
# --------------------------------------------------

async def initialize_mcp():
    """
    Discover MCP tools once and store them for reuse.
    """

    global search_tool
    global weather_tool
    global forecast_tool
    global aviation_tools
    global _all_tools

    if _all_tools is not None:
        return

    tools = await client.get_tools()

    # Store tools by name for easy lookup
    _all_tools = {
        tool.name: tool
        for tool in tools
    }

    print("\nAvailable MCP Tools:\n")

    for tool in tools:
        print(tool.name)

    # Tavily search tool
    search_tool = _all_tools.get("tavily_search")

    # Weather tools
    weather_tool = _all_tools.get("get_current_weather")
    forecast_tool = _all_tools.get("get_forcast")

    # Aviation tools only
    aviation_tools = {
        name: tool
        for name, tool in _all_tools.items()
        if name not in {
            "tavily_search",
            "tavily_extract",
            "tavily_crawl",
            "tavily_map",
            "tavily_research",
            "get_current_weather",
            "get_forcast",
        }
    }


def get_required_tool(tool_name: str):
    """
    Return a tool or raise a readable error.
    """

    if _all_tools is None:
        raise RuntimeError(
            "MCP tools are not initialized. "
            "Call initialize_mcp() first."
        )

    tool = _all_tools.get(tool_name)

    if tool is None:
        available = ", ".join(sorted(_all_tools.keys()))

        raise RuntimeError(
            f"MCP tool '{tool_name}' was not found.\n"
            f"Available tools: {available}"
        )

    return tool


# --------------------------------------------------
# Tavily MCP
# --------------------------------------------------

async def tavily_mcp_search(query: str):
    await initialize_mcp()

    if search_tool is None:
        raise RuntimeError(
            "Tavily search tool 'tavily_search' is unavailable."
        )

    result = await search_tool.ainvoke(
        {
            "query": query,
        }
    )

    return result


# --------------------------------------------------
# AviationStack MCP
# --------------------------------------------------

async def aviation_mcp_call(
    tool_name: str,
    tool_args: dict = None,
):
    await initialize_mcp()

    tool = aviation_tools.get(tool_name)

    if tool is None:
        raise RuntimeError(
            f"Aviation tool '{tool_name}' is unavailable."
        )

    result = await tool.ainvoke(
        tool_args or {}
    )

    return result


async def get_airports():
    await initialize_mcp()

    tool = aviation_tools.get("list_airports")

    if tool is None:
        raise RuntimeError(
            "Aviation tool 'list_airports' is unavailable."
        )

    return await tool.ainvoke({})


async def get_airlines():
    await initialize_mcp()

    tool = aviation_tools.get("list_airlines")

    if tool is None:
        raise RuntimeError(
            "Aviation tool 'list_airlines' is unavailable."
        )

    return await tool.ainvoke({})


# --------------------------------------------------
# Weather MCP
# --------------------------------------------------

async def initialize_weather_tools():
    await initialize_mcp()

    if weather_tool is None:
        raise RuntimeError(
            "Weather tool 'get_current_weather' is unavailable."
        )

    if forecast_tool is None:
        raise RuntimeError(
            "Weather tool 'get_forcast' is unavailable."
        )


async def weather_mcp_search(city: str):
    await initialize_weather_tools()

    return await weather_tool.ainvoke(
        {
            "city": city,
        }
    )


async def forecast_mcp_search(city: str):
    await initialize_weather_tools()

    return await forecast_tool.ainvoke(
        {
            "city": city,
        }
    )


# --------------------------------------------------
# Groq LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)


# --------------------------------------------------
# Destination Extractor
# --------------------------------------------------

def extract_destination(query: str) -> str:

    prompt = f"""
Extract only the destination city or country.

Query:
{query}

Return only the destination name.
"""

    response = llm.invoke(prompt)

    destination = response.content.strip()

    if not destination:
        raise ValueError(
            "The LLM returned an empty destination."
        )

    return destination


# --------------------------------------------------
# MCP Tool Discovery
# --------------------------------------------------

async def main():
    await initialize_mcp()


if __name__ == "__main__":
    asyncio.run(main())
from fastmcp import FastMCP

mcp = FastMCP("Weather", host="0.0.0.0", port=8080)

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    print("This is a log from the SSE Server")
    return "Hot as hell"


if __name__ == "__main__":
    mcp.run(transport="sse")
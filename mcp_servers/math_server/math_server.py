# math_server.py
from fastmcp import FastMCP

mcp = FastMCP("Math", host="0.0.0.0", port=8080)

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


if __name__ == "__main__":
    mcp.run(transport="sse")
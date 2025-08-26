import asyncio
from fastmcp import Client

async def test_plotly_graph():
    # Connect to your running Data Tool MCP server on port 9202
    client = Client("http://localhost:9202/sse/")
    async with client:
        # Call the tool with the desired argument
        result = await client.call_tool("get_sample_plotly_graph", {"graph_type": "bar"})
        print("Tool call result:")
        print(result)
        # Optionally, print just the HTML part
        if hasattr(result, 'structured_content') and 'html' in result.structured_content:
            print("\nHTML Output:\n")
            print(result.structured_content['html'])

if __name__ == "__main__":
    asyncio.run(test_plotly_graph())
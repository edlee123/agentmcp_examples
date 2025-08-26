
#!/usr/bin/env python3
"""
Data Tool MCP Server

This MCP server provides tools for data aggregation and analysis by calling the data-api service.
It offers various analytical functions that the LLM can use to perform calculations and insights
on the mock data.

Tools provided:
- get_user_stats: Get statistics about users
- get_product_analytics: Analyze product data (pricing, categories)
- get_document_insights: Analyze document content and tags
- search_and_analyze: Search data and provide analytical insights
- calculate_totals: Calculate various totals and averages
"""

import os 
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
import httpx
from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult

logging.basicConfig(level=logging.INFO)

log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

# Data API base URL (will be accessible via Docker network)
DATA_API_BASE_URL = "http://data-api:8000"

# Initialize FastMCP server
mcp = FastMCP("Data Tool Server", host="0.0.0.0", port=8080)


async def call_data_api(endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
    """Helper function to call the data-api service"""
    try:
        async with httpx.AsyncClient() as client:
            url = f"{DATA_API_BASE_URL}{endpoint}"
            response = await client.get(url, params=params or {})
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Error calling data API {endpoint}: {e}")
        return {"error": str(e)}

@mcp.tool()
async def get_user_stats(query_type: str) -> ToolResult:
    """
    Get comprehensive statistics about users in the system.
    
    Args:
        query_type: Type of query to perform (e.g., "all", "summary", "details")
        
    Returns:
        User statistics including:
        - Total number of users
        - Email domain distribution
        - User list with details
    """
    logger.info(f"get_user_stats called with query_type: {query_type!r}")
    try:
        # Get all users
        logger.info("Calling data API for users")
        users_data = await call_data_api("/users")
        
        if isinstance(users_data, dict) and "error" in users_data:
            return ToolResult(structured_content={"error": users_data["error"]})
        
        # Handle case where users_data is already a list
        users = users_data if isinstance(users_data, list) else users_data.get("users", [])
        
        # Calculate statistics
        total_users = len(users)
        
        # Email domain analysis
        email_domains = {}
        for user in users:
            email = user.get("email", "")
            if "@" in email:
                domain = email.split("@")[1]
                email_domains[domain] = email_domains.get(domain, 0) + 1
        
        # User names
        user_names = [user.get("name", "Unknown") for user in users]
        
        stats = {
            "total_users": total_users,
            "email_domains": email_domains,
            "user_names": user_names,
            "users_detail": users
        }
        
        return ToolResult(structured_content=stats)
        
    except Exception as e:
        logger.error(f"Error in get_user_stats: {e}")
        return ToolResult(structured_content={"error": str(e)})

@mcp.tool()
async def get_product_analytics(query_type: str) -> ToolResult:
    """
    Analyze product data including pricing, categories, and statistics.
    
    Args:
        query_type: Type of query to perform (e.g., "all", "summary", "details")
        
    Returns:
        Product analytics including:
        - Total products and categories
        - Price statistics (min, max, average)
        - Category distribution
        - Most/least expensive products
    """
    logger.info(f"get_product_analytics called with query_type: {query_type!r}")
    try:
        # Get all products
        logger.info("Calling data API for products")
        products_data = await call_data_api("/products")
        
        if isinstance(products_data, dict) and "error" in products_data:
            return ToolResult(structured_content={"error": products_data["error"]})
        
        # Handle case where products_data is already a list
        products = products_data if isinstance(products_data, list) else products_data.get("products", [])
        
        if not products:
            return ToolResult(structured_content={"message": "No products found"})
        
        # Extract prices and categories
        prices = [float(product.get("price", 0)) for product in products]
        categories = [product.get("category", "Unknown") for product in products]
        
        # Price statistics
        min_price = min(prices) if prices else 0
        max_price = max(prices) if prices else 0
        avg_price = sum(prices) / len(prices) if prices else 0
        
        # Category distribution
        category_counts = {}
        for category in categories:
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Find most and least expensive products
        most_expensive = max(products, key=lambda p: float(p.get("price", 0))) if products else None
        least_expensive = min(products, key=lambda p: float(p.get("price", 0))) if products else None
        
        analytics = {
            "total_products": len(products),
            "total_categories": len(category_counts),
            "price_statistics": {
                "minimum": min_price,
                "maximum": max_price,
                "average": round(avg_price, 2),
                "total_value": round(sum(prices), 2)
            },
            "category_distribution": category_counts,
            "most_expensive_product": most_expensive,
            "least_expensive_product": least_expensive,
            "products_detail": products
        }
        
        return ToolResult(structured_content=analytics)
        
    except Exception as e:
        logger.error(f"Error in get_product_analytics: {e}")
        return ToolResult(structured_content={"error": str(e)})

@mcp.tool()
async def get_document_insights(query_type: str) -> ToolResult:
    """
    Analyze document data including content analysis and tag statistics.
    
    Args:
        query_type: Type of query to perform (e.g., "all", "summary", "tags")
        
    Returns:
        Document insights including:
        - Total documents
        - Tag frequency analysis
        - Content length statistics
        - Most common tags
    """
    logger.info(f"get_document_insights called with query_type: {query_type!r}")
    try:
        # Get all documents
        logger.info("Calling data API for documents")
        documents_data = await call_data_api("/documents")
        
        if isinstance(documents_data, dict) and "error" in documents_data:
            return ToolResult(structured_content={"error": documents_data["error"]})
        
        # Handle case where documents_data is already a list
        documents = documents_data if isinstance(documents_data, list) else documents_data.get("documents", [])
        
        if not documents:
            return ToolResult(structured_content={"message": "No documents found"})
        
        # Analyze tags
        all_tags = []
        for doc in documents:
            tags = doc.get("tags", [])
            all_tags.extend(tags)
        
        # Tag frequency
        tag_frequency = {}
        for tag in all_tags:
            tag_frequency[tag] = tag_frequency.get(tag, 0) + 1
        
        # Content length analysis
        content_lengths = []
        for doc in documents:
            content = doc.get("content", "")
            content_lengths.append(len(content))
        
        # Calculate content statistics
        avg_content_length = sum(content_lengths) / len(content_lengths) if content_lengths else 0
        min_content_length = min(content_lengths) if content_lengths else 0
        max_content_length = max(content_lengths) if content_lengths else 0
        
        # Find most common tags
        sorted_tags = sorted(tag_frequency.items(), key=lambda x: x[1], reverse=True)
        
        insights = {
            "total_documents": len(documents),
            "total_unique_tags": len(tag_frequency),
            "tag_frequency": tag_frequency,
            "most_common_tags": sorted_tags[:5],  # Top 5 tags
            "content_statistics": {
                "average_length": round(avg_content_length, 2),
                "minimum_length": min_content_length,
                "maximum_length": max_content_length,
                "total_content_length": sum(content_lengths)
            },
            "documents_detail": documents
        }
        
        return ToolResult(structured_content=insights)
        
    except Exception as e:
        logger.error(f"Error in get_document_insights: {e}")
        return ToolResult(structured_content={"error": str(e)})

@mcp.tool()
async def search_and_analyze(query: str, data_type: Optional[str] = None) -> ToolResult:
    """
    Search the data and provide analytical insights on the results.
    
    Args:
        query: Search term to look for
        data_type: Optional filter for data type ("users", "products", "documents")
    
    Returns:
        Search results and analysis
    """
    logger.info(f"search_and_analyze called with query: {query!r}, data_type: {data_type!r}")
    try:
        # Perform search
        params = {"q": query}
        if data_type:
            params["type"] = data_type
        
        search_data = await call_data_api("/search", params)
        
        if isinstance(search_data, dict) and "error" in search_data:
            return ToolResult(structured_content={"error": search_data["error"]})
        
        # Handle case where search_data is already a list
        results = search_data if isinstance(search_data, list) else search_data.get("results", [])
        
        # Analyze search results
        result_types = {}
        for result in results:
            result_type = result.get("type", "unknown")
            result_types[result_type] = result_types.get(result_type, 0) + 1
        
        # Extract specific insights based on result types
        insights = {
            "search_query": query,
            "total_results": len(results),
            "result_type_distribution": result_types,
            "search_metadata": search_data.get("search_info", {}),
            "results": results
        }
        
        # Add type-specific analysis
        if "user" in result_types:
            user_results = [r["data"] for r in results if r["type"] == "user"]
            insights["user_analysis"] = {
                "count": len(user_results),
                "emails": [u.get("email") for u in user_results]
            }
        
        if "product" in result_types:
            product_results = [r["data"] for r in results if r["type"] == "product"]
            prices = [float(p.get("price", 0)) for p in product_results]
            insights["product_analysis"] = {
                "count": len(product_results),
                "total_value": round(sum(prices), 2),
                "average_price": round(sum(prices) / len(prices), 2) if prices else 0
            }
        
        if "document" in result_types:
            doc_results = [r["data"] for r in results if r["type"] == "document"]
            all_tags = []
            for doc in doc_results:
                all_tags.extend(doc.get("tags", []))
            insights["document_analysis"] = {
                "count": len(doc_results),
                "unique_tags": list(set(all_tags)),
                "tag_count": len(all_tags)
            }
        
        return ToolResult(structured_content=insights)
        
    except Exception as e:
        logger.error(f"Error in search_and_analyze: {e}")
        return ToolResult(structured_content={"error": str(e)})

@mcp.tool()
async def calculate_totals(query_type: str) -> ToolResult:
    """
    Calculate various totals and aggregations across all data types.
    
    Args:
        query_type: Type of query to perform (e.g., "all", "financial", "content")
        
    Returns:
        Comprehensive totals and calculations
    """
    logger.info(f"calculate_totals called with query_type: {query_type!r}")
    try:
        # Get all data types
        logger.info("Calling data API for users, products, and documents")
        users_data = await call_data_api("/users")
        products_data = await call_data_api("/products")
        documents_data = await call_data_api("/documents")
        
        # Check for errors
        errors = []
        for data in [users_data, products_data, documents_data]:
            if isinstance(data, dict) and "error" in data:
                errors.append(data["error"])
        
        if errors:
            return ToolResult(structured_content={"errors": errors})
        
        # Handle case where data is already a list
        users = users_data if isinstance(users_data, list) else users_data.get("users", [])
        products = products_data if isinstance(products_data, list) else products_data.get("products", [])
        documents = documents_data if isinstance(documents_data, list) else documents_data.get("documents", [])
        
        # Calculate totals
        total_users = len(users)
        total_products = len(products)
        total_documents = len(documents)
        
        # Product calculations
        product_prices = [float(p.get("price", 0)) for p in products]
        total_product_value = sum(product_prices)
        average_product_price = total_product_value / len(product_prices) if product_prices else 0
        
        # Document calculations
        total_content_length = sum(len(doc.get("content", "")) for doc in documents)
        all_document_tags = []
        for doc in documents:
            all_document_tags.extend(doc.get("tags", []))
        unique_tags = list(set(all_document_tags))
        
        # Email domain calculations
        email_domains = set()
        for user in users:
            email = user.get("email", "")
            if "@" in email:
                email_domains.add(email.split("@")[1])
        
        # Product category calculations
        product_categories = set(p.get("category", "") for p in products)
        
        totals = {
            "data_summary": {
                "total_users": total_users,
                "total_products": total_products,
                "total_documents": total_documents,
                "total_records": total_users + total_products + total_documents
            },
            "financial_summary": {
                "total_product_value": round(total_product_value, 2),
                "average_product_price": round(average_product_price, 2),
                "most_expensive_price": max(product_prices) if product_prices else 0,
                "least_expensive_price": min(product_prices) if product_prices else 0
            },
            "content_summary": {
                "total_content_characters": total_content_length,
                "total_unique_tags": len(unique_tags),
                "total_tag_instances": len(all_document_tags),
                "unique_tags_list": unique_tags
            },
            "diversity_summary": {
                "unique_email_domains": len(email_domains),
                "email_domains_list": list(email_domains),
                "unique_product_categories": len(product_categories),
                "product_categories_list": list(product_categories)
            }
        }
        
        return ToolResult(structured_content=totals)
        
    except Exception as e:
        logger.error(f"Error in calculate_totals: {e}")
        return ToolResult(structured_content={"error": str(e)})

@mcp.tool()
async def get_data_health_check(check_type: str) -> ToolResult:
    """
    Perform a health check on the data-api service and return connectivity status.
    
    Args:
        check_type: Type of health check to perform (e.g., "basic", "full", "endpoints")
        
    Returns:
        Health check results
    """
    logger.info(f"get_data_health_check called with check_type: {check_type!r}")
    try:
        # Test connectivity to data-api
        logger.info("Calling data API for health check")
        health_data = await call_data_api("/health")
        
        if isinstance(health_data, dict) and "error" in health_data:
            return ToolResult(structured_content={
                "status": "unhealthy",
                "error": health_data["error"],
                "data_api_url": DATA_API_BASE_URL
            })
        
        # Test each endpoint
        endpoints_status = {}
        test_endpoints = ["/users", "/products", "/documents", "/search?q=test"]
        
        for endpoint in test_endpoints:
            try:
                result = await call_data_api(endpoint)
                endpoints_status[endpoint] = "healthy" if not (isinstance(result, dict) and "error" in result) else "error"
            except Exception as e:
                endpoints_status[endpoint] = f"error: {str(e)}"
        
        health_check = {
            "status": "healthy",
            "data_api_url": DATA_API_BASE_URL,
            "data_api_health": health_data,
            "endpoints_status": endpoints_status,
            "timestamp": health_data.get("timestamp", "unknown")
        }
        
        return ToolResult(structured_content=health_check)
        
    except Exception as e:
        logger.error(f"Error in get_data_health_check: {e}")
        return ToolResult(structured_content={
            "status": "unhealthy",
            "error": str(e),
            "data_api_url": DATA_API_BASE_URL
        })
    

# =====================
# Plotly Graph Tool
# =====================
import plotly.graph_objs as go

@mcp.tool()
async def get_sample_plotly_graph(graph_type: str = "bar") -> ToolResult:
    """
    Generate a sample Plotly graph and return it as HTML for rendering in OpenWebUI.
    Args:
        graph_type: Type of graph to generate ("bar", "line", "scatter")
    Returns:
        HTML string containing the Plotly graph
    """
    logger.info(f"get_sample_plotly_graph called with graph_type: {graph_type!r}")
    try:
        # Example data
        x = ["A", "B", "C", "D"]
        y = [10, 15, 7, 12]

        if graph_type == "bar":
            fig = go.Figure([go.Bar(x=x, y=y)])
            fig.update_layout(title="Sample Bar Chart")
        elif graph_type == "line":
            fig = go.Figure([go.Scatter(x=x, y=y, mode="lines+markers")])
            fig.update_layout(title="Sample Line Chart")
        elif graph_type == "scatter":
            fig = go.Figure([go.Scatter(x=x, y=y, mode="markers")])
            fig.update_layout(title="Sample Scatter Plot")
        else:
            return ToolResult(structured_content={"error": f"Unknown graph_type: {graph_type}"})

        # Generate HTML
        html = fig.to_html(full_html=True, include_plotlyjs="cdn")
        logger.info(f"Generated Plotly HTML (truncated to 500 chars):\n{html[:1000]}\n...")
        # return ToolResult(structured_content={"html": html, "graph_type": graph_type})
        return ToolResult(content=html)
    except Exception as e:
        logger.error(f"Error in get_sample_plotly_graph: {e}")
        return ToolResult(structured_content={"error": str(e)})
    

# =====================
# Sample HTML Artifact Tool
# =====================
@mcp.tool()
async def sample_html_artifact() -> ToolResult:
    """
    Return a simple HTML page as an artifact for OpenWebUI rendering.
    This demonstrates the correct format for HTML artifact output.
    """
    logger.info("sample_html_artifact called")
    html = """<!DOCTYPE html>
<html>
  <head>
    <meta charset='utf-8'>
    <title>Sample Artifact</title>
    <style>
      body { font-family: sans-serif; margin: 2em; background: #f9f9f9; }
      .artifact-box { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 2em; box-shadow: 0 2px 8px #0001; }
      h1 { color: #2a3f5f; }
    </style>
  </head>
  <body>
    <div class='artifact-box'>
      <h1>OpenWebUI Artifact Demo</h1>
      <p>This is a <b>sample HTML artifact</b> returned by an MCP tool.<br>
      You should see this rendered as a full web page in OpenWebUI.</p>
      <ul>
        <li>Supports <b>HTML</b> and <b>CSS</b></li>
        <li>Can include images, charts, etc.</li>
      </ul>
    </div>
  </body>
</html>"""
    logger.info("Returning sample HTML artifact.")
    return ToolResult(content=html)

if __name__ == "__main__":
    # Run the MCP server
    mcp.run(transport="sse")

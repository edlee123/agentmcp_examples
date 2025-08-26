"""
Orchestration API for OpenWebUI Integration

This service acts as a bridge between OpenWebUI (a ChatGPT-like web interface) and 
a microservices architecture that includes:
- LangChain agents with tool calling capabilities
- Multiple MCP (Model Context Protocol) servers providing specialized tools
- Streaming chat completions with real-time responses

Architecture Overview:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   OpenWebUI     │───▶│ Orchestration   │───▶│   MCP Servers   │
│ (Web Interface) │    │      API        │    │ (Tools/Services)│
│                 │◀───│  (This Service) │◀───│                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   OpenRouter    │
                       │ (LLM Provider)  │
                       └─────────────────┘

Key Concepts:
- MCP (Model Context Protocol): Allows LLMs to securely connect to external tools/data
- LangChain Agents: Intelligent systems that can reason about and use multiple tools
- Streaming Responses: Real-time token-by-token response delivery for better UX
- OpenAI API Compatibility: Standard format that OpenWebUI expects
"""

import time
import sys
import os
import asyncio
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
import httpx
import yaml

# Import response handlers and models
from response_handlers import (
    stream_api_key_warning,
    stream_chat_completion,
    non_stream_chat_completion,
    test_tool_calling_capability,
    ChatMessage,
    ChatCompletionRequest
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG to see more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Orchestration API (OpenWebUI)", description="Coordinates LLM and MCP services with streaming support")

# Get model from environment variable

# Load SUPPORTED_MODELS from config.yaml
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

DEFAULT_LLM_MODEL = next(iter(CONFIG["supported_models"])) 

# Import multi-provider configuration
from models import (
    get_available_models,
    is_model_available,
    create_llm_for_model
)


# =============================================================================
# 🧪 STARTUP TOOL CALLING CAPABILITY TEST
# =============================================================================

# Global variable to store tool calling capability test result
LLM_SUPPORTS_TOOL_CALLING = None

@app.on_event("startup")
async def startup_event():
    """Initialize the application and validate model configuration"""
    logger.info("🚀 Starting Orchestration API...")
    
    # Get available models
    available_models = get_available_models()
    
    if not available_models:
        logger.error("❌ No models available. Please check your API keys and configuration.")
        sys.exit(1)
    
    logger.info(f"✅ Available models: {', '.join(available_models)}")
    
    # Set up globals based on first available model
    global LLM_MODEL, LLM_SUPPORTS_TOOL_CALLING
    first_model = available_models[0]
    LLM_MODEL = first_model
    
    # Test tool calling capability for the first model
    test_llm = create_llm_for_model(first_model)
    
    try:
        start_time = time.time()
        LLM_SUPPORTS_TOOL_CALLING = await test_tool_calling_capability(test_llm, first_model)
        test_duration = time.time() - start_time
        logger.info(f"🔧 Tool calling test completed in {test_duration:.3f}s for {first_model}")
    except Exception as e:
        logger.error(f"⚠️ Startup tool calling test failed: {e}")
        LLM_SUPPORTS_TOOL_CALLING = False

# CORS middleware configuration for OpenWebUI.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# 🚀 API ENDPOINTS
# =============================================================================

@app.get("/v1/models")
async def list_models():
    """
    OpenAI-compatible models endpoint - REQUIRED for OpenWebUI integration
    
    Returns list of available models in OpenAI format.
    OpenWebUI uses this to populate the model dropdown.
    """
    available_models = get_available_models()
    
    models = []
    for model_id in available_models:
        config = CONFIG["supported_models"].get(model_id, {})
        models.append({
            "id": model_id,
            "object": "model",
            "created": 1677610602,
            "owned_by": config.get("provider", "openrouter"),
            "permission": [],
            "root": model_id,
            "parent": None
        })
    
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, chat_request: ChatCompletionRequest):
    """
    Main chat endpoint - OpenWebUI sends all chat messages here
    
    OpenWebUI Integration Flow:
    1. User types message in OpenWebUI chat interface
    2. OpenWebUI formats request as OpenAI-compatible payload
    3. Request hits this endpoint with user message + history
    4. If no API key: Returns streaming warning (user sees error)
    5. If valid key: Routes to LangGraph agent for processing
    6. Agent uses MCP servers (weather/math/data tools) to gather info
    7. Streams back: [Agent reasoning] → [Tool calls] → [Final answer]
    8. OpenWebUI displays streamed response in real-time
    
    MCP Server Integration:
    - Weather queries → weather_server (local conditions)
    - Math calculations → math_server (complex computations)  
    - Data operations → data_tool_server (file/database access)
    
    Args:
        request: FastAPI request object (for IP logging)
        chat_request: Pydantic model with messages, model, stream settings
        
    Returns:
        
    Example Request:
        {
            "model": "anthropic/claude-3.5-sonnet",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "stream": true
        }
    """
    # Log the incoming request for debugging
    logger.info(f" Request model: {chat_request.model}")
    logger.info(f"🔗 Request stream: {chat_request.stream}")
    logger.info(f"📊 Message count: {len(chat_request.messages)}")

    # Check if API key is provided and valid
    api_key = request.headers.get("authorization", "").replace("Bearer ", "")
    logger.info(f"🔑 API key received: {api_key[:10]}..." if api_key else "🚫 No API key found")
    logger.info(f"🔍 API key validation: {api_key not in ['', 'sk-xxx', None]}")

    # Create LLM instance for the requested model
    try:
        llm_streaming = create_llm_for_model(chat_request.model)
        logger.info(f"✅ Created LLM instance for {chat_request.model}")
    except Exception as e:
        logger.error(f"❌ Failed to create LLM for {chat_request.model}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create LLM instance: {e}")

    if not api_key or api_key in ["", "sk-xxx"]:
        logger.warning("❌ API key validation failed - returning warning")
        return await stream_api_key_warning(chat_request.model)

    if not is_model_available(chat_request.model):
        raise HTTPException(
            status_code=400,
            detail=f"Model {chat_request.model} is not available. Use /v1/models to see available models."
        )

    # Extract the latest user message
    user_message = ""
    for message in reversed(chat_request.messages):
        if message.role == "user":
            user_message = message.content
            break

    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found in request")

    # Generate response based on request type
    if chat_request.stream:
        logger.info("🌊 Generating streaming response...")
        return await stream_chat_completion(
            user_message, 
            chat_request.model, 
            get_mcp_client, 
            get_system_message, 
            llm_streaming
        )
    else:
        logger.info("📦 Generating non-streaming response...")
        return await non_stream_chat_completion(
            user_message, 
            chat_request.model, 
            get_mcp_client, 
            get_system_message, 
            llm_streaming
        )
    

@app.get("/v1/tools")
async def list_tools():
    """
    Debug endpoint to list available MCP tools
    This helps diagnose MCP client connectivity issues
    """
    try:
        async with await get_mcp_client() as client:
            tools = [tool for tool in client.get_tools()]
            
            tool_list = []
            for tool in tools:
                tool_info = {
                    "name": getattr(tool, 'name', 'unknown'),
                    "type": type(tool).__name__,
                    "description": getattr(tool, 'description', 'no description')
                }
                tool_list.append(tool_info)
            
            return {
                "status": "success",
                "tool_count": len(tools),
                "tools": tool_list,
                "mcp_client_type": type(client).__name__
            }
    except Exception as e:
        logger.error(f"Error listing tools: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "tool_count": 0,
            "tools": []
        }



# =============================================================================
# ⚙️ UTILITY FUNCTIONS (Configuration & Helpers)
# =============================================================================


def get_system_message():
    """System message for the agent"""
    return """You are a helpful assistant with access to several tools.
    ALWAYS use the appropriate tool when asked about:
    - Weather information (use the weather tool)
    - Mathematical calculations (use the math tool)
    - User statistics (use the get_user_stats tool)
    - Product analytics (use the get_product_analytics tool)
    - Document insights (use the get_document_insights tool)
    - Data search and analysis (use the search_and_analyze tool)
    - Calculating totals (use the calculate_totals tool)
    
    When asked about any of these topics, you MUST use the relevant tool rather than making up information.
    
    IMPORTANT: After deciding to use a tool, you MUST execute it by calling the tool function.
    DO NOT just say you will use the tool - actually use it!
    
    CRITICAL: When calling a tool, you MUST provide valid JSON arguments with the required parameters."""


@app.get("/health")
async def health_check():
    """
    Health check endpoint for container orchestration and monitoring
    Reports unhealthy if model validation failed or no models available
    """
    try:
        # Import validation status
        from models import STARTUP_VALIDATION_COMPLETE, VALIDATED_MODELS
        
        # Check if startup validation completed
        if not STARTUP_VALIDATION_COMPLETE:
            logger.warning("⚠️ Health check: Model validation not complete yet")
            raise HTTPException(
                status_code=503, 
                detail="Model validation in progress - service not ready"
            )
        
        # Check if any models passed validation
        available_models = get_available_models()
        model_count = len(available_models)
        
        if model_count == 0:
            logger.error("❌ Health check: No validated models available")
            raise HTTPException(
                status_code=503,
                detail="No working models available - service unhealthy"
            )
        
        # Check if the default model is working
        default_model_available = is_model_available(DEFAULT_LLM_MODEL)
        if not default_model_available:
            logger.error(f"❌ Health check: Default model {DEFAULT_LLM_MODEL} not available")
            raise HTTPException(
                status_code=503,
                detail=f"Default model {DEFAULT_LLM_MODEL} failed validation - service unhealthy"
            )
        
        # All checks passed - service is healthy
        logger.info(f"✅ Health check passed: {model_count} models available, default model working")
        
        return {
            "status": "healthy",
            "message": "Orchestration API is running",
            "available_models": model_count,
            "validated_models": list(available_models),
            "tool_calling_enabled": LLM_SUPPORTS_TOOL_CALLING,
            "default_model": DEFAULT_LLM_MODEL,
            "validation_complete": STARTUP_VALIDATION_COMPLETE
        }
    except HTTPException:
        # Re-raise HTTP exceptions (they already have proper status codes)
        raise
    except Exception as e:
        logger.error(f"❌ Health check failed with unexpected error: {e}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {e}")


@app.get("/")
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "🤖 Orchestration API for OpenWebUI",
        "version": "1.0.0",
        "endpoints": {
            "models": "/v1/models",
            "chat": "/v1/chat/completions",
            "health": "/health"
        },
        "documentation": "/docs"
    }

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & UTILITY FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

async def get_mcp_client():
    """
    Initialize and return MCP (Model Context Protocol) client with all servers from config.yaml
    Returns:
        MultiServerMCPClient: Configured client with all MCP servers
    """
    return MultiServerMCPClient(CONFIG["mcp_servers"])

def get_system_message():
    """
    Get the system message that instructs the LLM on tool usage
    
    This message encourages the agent to:
    - Use appropriate tools for specific queries
    - Provide valid JSON arguments
    - Execute tools rather than just describing them
    
    Returns:
        str: System message for the agent
    """
    return """You are a helpful assistant with access to several tools.
    ALWAYS use the appropriate tool when asked about:
    - Weather information (use the weather tool)
    - Mathematical calculations (use the math tool)
    - User statistics (use the get_user_stats tool)
    - Product analytics (use the get_product_analytics tool)
    - Document insights (use the get_document_insights tool)
    - Data search and analysis (use the search_and_analyze tool)
    - Calculating totals (use the calculate_totals tool)
    
    When asked about any of these topics, you MUST use the relevant tool rather than making up information.
    
    IMPORTANT: After deciding to use a tool, you MUST execute it by calling the tool function.
    DO NOT just say you will use the tool - actually use it!
    
    CRITICAL: When calling a tool, you MUST provide valid JSON arguments with the required parameters.
    For example:
    - get_user_stats: {"query_type": "all"}
    - get_product_analytics: {"query": "all"}
    - get_document_insights: {"dummy": 1}
    - calculate_totals: {"dummy": 1}
    - get_data_health_check: {"dummy": 1}
    
    ALWAYS include the required parameters for each tool. The get_user_stats tool REQUIRES the query_type parameter."""

# ═════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/chat")
async def chat(message_data: dict):
    """
    Main chat endpoint for Gradio UI integration (non-streaming)
    
    This endpoint:
    1. Accepts user messages from Gradio interface
    2. Delegates to response_handlers.simple_chat_completion
    3. Returns complete response with tool call details
    
    Args:
        message_data: Dictionary containing user message
        
    Returns:
        dict: Response with agent output and tool call details
        
    Example:
        Request: {"message": "What's 15 * 25?"}
        Response: {"response": "15 * 25 = 375", "tool_calls": [...]}
    """
    message = message_data.get("message", "")
    
    # Log the incoming request
    logger.info(f"💬 Chat request: {message[:100]}...")
    
    # Check API key
    if api_key == "dummy-key":
        return {
            "response": "⚠️ **API Key Required**: Please set OPENAI_API_KEY in your environment variables to use this chat interface.",
            "tool_calls": []
        }
    
    try:
        # Use the imported simple_chat_completion function
        result = await simple_chat_completion(
            message=message,
            get_mcp_client_func=get_mcp_client,
            get_system_message_func=get_system_message,
            llm_non_streaming=llm_non_streaming
        )
        return result
            
    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

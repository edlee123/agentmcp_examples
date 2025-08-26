"""
Response handlers for OpenAI-compatible chat completions.

This module contains all the streaming and non-streaming response generation logic,
plus the Pydantic models for request/response validation,
separated from the main API routing for better organization and maintainability.
"""
import os
import logging

import asyncio
import json
import logging
from fastapi.responses import StreamingResponse
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, AsyncGenerator, Generator

log_level = os.environ.get("LOGGFLAG", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

# =============================================================================
# 🧪 TOOL CALLING CAPABILITY TEST
# =============================================================================

async def test_tool_calling_capability(llm, model: str) -> bool:
    """
    Test if the LLM supports tool calling (fail fast)
    
    Args:
        llm: The LangChain LLM instance to test
        model: Model identifier for logging
        
    Returns:
        bool: True if tool calling is supported, False otherwise
    """
    logger.info(f"🧪 Testing tool calling capability for {model}...")
    
    try:
        # Create a simple test function
        test_tool = {
            "type": "function",
            "function": {
                "name": "test_capability",
                "description": "A test function to check capability",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "test": {"type": "string", "description": "Test parameter"}
                    },
                    "required": ["test"]
                }
            }
        }
        
        # Try to bind the tool to the LLM. bind_tools() creates NEW instance, doesn't modify original
        llm_with_tools = llm.bind_tools([test_tool])
        
        # Make a simple test call
        response = await llm_with_tools.ainvoke(
            "Test if you can call the test_capability function with parameter 'hello'"
        )
        
        # Check if the response contains tool calls
        supports_tool_calling = False
        if hasattr(response, 'tool_calls') and response.tool_calls:
            supports_tool_calling = True
        elif hasattr(response, 'additional_kwargs') and 'tool_calls' in response.additional_kwargs:
            supports_tool_calling = True
        
        if supports_tool_calling:
            logger.info(f"✅ {model} supports tool calling")
        else:
            logger.warning(f"❌ {model} does not support tool calling")
        
        return supports_tool_calling
        
    except Exception as e:
        logger.warning(f"⚠️ Tool calling test failed for {model}: {e}")
        return False

# =============================================================================
# 📋 PYDANTIC MODELS (Request/Response Validation)
# =============================================================================

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "anthropic/claude-3.5-sonnet"  # Default model
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = None

# =============================================================================
# 🔧 MAIN RESPONSE HANDLERS (Public API)
# =============================================================================

# Standard headers for streaming responses
STREAMING_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Content-Type": "text/event-stream",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "*",
}


async def stream_chat_completion(message: str, model: str, get_mcp_client_func, get_system_message_func, llm_streaming) -> StreamingResponse:
    """Return streaming chat completion response"""
    return StreamingResponse(
        _generate_streaming_response(message, model, get_mcp_client_func, get_system_message_func, llm_streaming),
        media_type="text/event-stream",
        headers=STREAMING_HEADERS
    )


async def simple_chat_completion(message: str, get_mcp_client_func, get_system_message_func, llm_non_streaming) -> Dict[str, str]:
    """
    Simple chat completion for Gradio UI integration (non-streaming)
    
    This function:
    - Connects to MCP servers to get available tools
    - Creates a LangChain ReAct agent with tools  
    - Executes the agent and returns response with tool call info
    - Designed for simple request/response patterns
    
    Args:
        message: User input message
        get_mcp_client_func: Function to get MCP client
        get_system_message_func: Function to get system message
        llm_non_streaming: Non-streaming LLM instance
        
    Returns:
        dict: Simple response with content and tool call details
    """
    logger.info(f"💬 Processing simple chat: {message[:100]}...")
    
    try:
        async with await get_mcp_client_func() as client:
            # Get available tools from MCP servers
            tools = [tool for tool in client.get_tools()]
            logger.info(f"🔧 Available tools: {[tool.name for tool in tools]}")
            
            # Create ReAct agent with tools
            agent = create_react_agent(
                llm_non_streaming,
                tools,
                prompt=get_system_message_func()
            )
            
            # Format message for agent
            formatted_message = [{"role": "user", "content": message}]
            
            # Execute agent
            logger.info(f"🤖 Executing agent...")
            result = await agent.ainvoke({"messages": formatted_message})
            
            # Enhanced debugging using shared logging function
            if isinstance(result, dict) and "messages" in result:
                _log_agent_messages(result["messages"], "Agent")
            
            logger.info(f"✅ Agent execution completed successfully")
            
            return {
                "response": result["messages"][-1].content
            }
            
    except Exception as e:
        logger.error(f"❌ Error in simple chat completion: {str(e)}")
        raise


async def _generate_streaming_response(message: str, model: str, get_mcp_client_func, get_system_message_func, llm_streaming) -> AsyncGenerator[str, None]:
    """Internal generator for streaming response content"""
    try:
        async with await get_mcp_client_func() as client:
            tools = [tool for tool in client.get_tools()]
            logger.info(f"Available tools for streaming: {len(tools)}")
            
            agent = create_react_agent(
                llm_streaming,
                tools,
                prompt=get_system_message_func()
            )
            
            formatted_message = [{"role": "user", "content": message}]
            logger.info(f"Starting LangGraph streaming for message: {message}")
            
            chunk_id = f"chatcmpl-{hash(message) & 0x7fffffff:08x}"
            has_streamed_content = False
            
            logger.info("About to start LangGraph streaming with stream_mode='values'")
            last_message_count = 0
            
            async for state in agent.astream({"messages": formatted_message}, stream_mode="values"):
                logger.info(f"LangGraph state received: {type(state)} with keys: {state.keys() if isinstance(state, dict) else 'N/A'}")
                
                if isinstance(state, dict) and "messages" in state:
                    messages = state["messages"]
                    logger.info(f"State contains {len(messages)} messages")
                    
                    # Use shared logging function for consistent debugging
                    _log_agent_messages(messages, "State")
                    
                    # Stream new messages using extracted helper function
                    for chunk in _stream_new_messages(messages, last_message_count, chunk_id, model):
                        has_streamed_content = True
                        yield chunk
                    
                    last_message_count = len(messages)
            
            # If LangGraph streaming didn't work, log an error
            if not has_streamed_content:
                logger.error("LangGraph streaming failed to yield any content - true streaming not working")
                
                # Send error message as a single chunk using helper function
                error_content = "Error: Streaming failed to produce content."
                stream_chunk = _create_stream_chunk(chunk_id, model, error_content)
                yield f"data: {json.dumps(stream_chunk)}\n\n"
            
            # Send final chunk using helper function
            final_chunk = _create_stream_chunk(chunk_id, model, "", "stop")
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            
    except Exception as e:
        logger.error(f"Error in LangGraph streaming: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Use helper function for consistent error chunk creation
        error_chunk = _create_error_chunk(str(e))
        yield f"data: {json.dumps(error_chunk)}\n\n"


# =============================================================================
# 🛠️ HELPER FUNCTIONS (Private Utilities)
# =============================================================================

def _log_agent_messages(messages: list, context: str = "Agent") -> None:
    """
    Shared logging function for debugging agent messages and tool calls
    
    Args:
        messages: List of messages from agent execution
        context: Context string for logging (e.g. "Agent", "State")
    """
    logger.info(f"{context} execution completed with {len(messages)} messages")
    
    # Log details about each message type for debugging
    for idx, msg in enumerate(messages):
        msg_type = type(msg).__name__
        if hasattr(msg, 'type'):
            msg_type = f"{msg_type}({msg.type})"
        
        if hasattr(msg, 'content'):
            content_preview = msg.content[:100] if msg.content else "No content"
            logger.debug(f"Message {idx}: {msg_type} - {content_preview}")
        
        # Specific logging for tool calls
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            logger.info(f"🔧 TOOL CALL detected in message {idx}: {len(msg.tool_calls)} calls")
            for tool_idx, tool_call in enumerate(msg.tool_calls):
                # Debug: log the actual structure of tool_call
                logger.debug(f"Tool call structure: {type(tool_call)} - {tool_call}")
                
                if isinstance(tool_call, dict):
                    # Check multiple possible locations for tool name
                    tool_name = (tool_call.get('name') or 
                                tool_call.get('function', {}).get('name') or 
                                'unknown')
                elif hasattr(tool_call, 'function') and hasattr(tool_call.function, 'name'):
                    tool_name = tool_call.function.name
                elif hasattr(tool_call, 'name'):
                    tool_name = tool_call.name
                else:
                    tool_name = f"unknown ({type(tool_call).__name__})"
                
                logger.info(f"  Tool {tool_idx}: {tool_name}")
        
        # Log if this is a tool response/result
        if hasattr(msg, 'name') and msg.name:
            logger.info(f"🔨 TOOL RESULT from {msg.name}: {content_preview}")


def _create_stream_chunk(chunk_id: str, model: str, content: str, finish_reason: str = None) -> Dict[str, Any]:
    """
    Create a standard OpenAI-compatible streaming chunk
    
    Args:
        chunk_id: Unique identifier for this chat completion
        model: Model name for the response
        content: Content to include in the delta (empty string for final chunks)
        finish_reason: Finish reason for the chunk (None for content chunks, "stop" for final)
        
    Returns:
        dict: Formatted streaming chunk
    """
    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(asyncio.get_event_loop().time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason
        }]
    }
    
    # Add content to delta if provided
    if content:
        chunk["choices"][0]["delta"]["content"] = content
    
    return chunk


def _create_error_chunk(error_message: str, error_type: str = "internal_error") -> Dict[str, Any]:
    """
    Create a standard error chunk for streaming responses
    
    Args:
        error_message: Error message to include
        error_type: Type of error (default: "internal_error")
        
    Returns:
        dict: Formatted error chunk
    """
    return {
        "error": {
            "message": error_message,
            "type": error_type
        }
    }


def _stream_new_messages(messages: list, last_message_count: int, chunk_id: str, model: str) -> Generator[str, None, None]:
    """
    Stream any new AI message chunks since the last update
    
    Args:
        messages: Current list of messages from agent state
        last_message_count: Number of messages processed in previous iteration
        chunk_id: Unique identifier for this chat completion
        model: Model name for the response
        
    Yields:
        str: Formatted SSE chunks for new AI messages
    """
    # Stream any new messages since our last update
    if len(messages) > last_message_count:
        for i in range(last_message_count, len(messages)):
            message_obj = messages[i]
            if hasattr(message_obj, 'content') and message_obj.content and message_obj.content.strip():
                content = message_obj.content
                message_type = type(message_obj).__name__
                
                # Only stream AI messages (reasoning and final responses), skip tool messages
                if message_type == "AIMessage":
                    logger.info(f"📤 Streaming message {i} ({message_type}): {content[:100]}...")
                    
                    # Stream this AI message content using helper function
                    stream_chunk = _create_stream_chunk(chunk_id, model, content + "\n\n")
                    yield f"data: {json.dumps(stream_chunk)}\n\n"
                else:
                    logger.debug(f"⏭️ Skipping message {i} ({message_type}): {content[:50]}...")


# Standard headers for streaming responses
STREAMING_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Content-Type": "text/event-stream",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "*",
}


async def stream_api_key_warning(model: str) -> StreamingResponse:
    """Return API key warning as streaming response e.g. for UI"""
    async def generate():
        warning_content = """⚠️ **API Key Required**: To use this chat interface, you need to provide an API key.

**Secure Setup Options:**

**Method 1 (Recommended)**: Environment variables in terminal
```bash
export OPENAI_API_KEY="your_actual_key"
docker-compose up -d
```

**Method 2**: Use .env file for development only
1. Get an API key from https://openrouter.ai/keys (or OpenAI/other providers)
2. Copy `.env.example` to `.env` and add your key
3. Restart the containers

**Multiple Provider Support:**
- OpenRouter: `OPENAI_API_KEY=sk-or-...` (supports many models)
- OpenAI: `OPENAI_API_KEY=sk-...` + set `OPENROUTER_BASE_URL=https://api.openai.com/v1`
- Others: Set appropriate base URL

⚠️ **Security**: Never commit API keys to version control!

I can still help you with tool demonstrations once you set up the API key!"""
        
        # Use helper function for consistent chunk creation
        stream_chunk = _create_stream_chunk("chatcmpl-nokey001", model, warning_content)
        yield f"data: {json.dumps(stream_chunk)}\n\n"
        
        # Send final chunk using helper function
        final_chunk = _create_stream_chunk("chatcmpl-nokey001", model, "", "stop")
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=STREAMING_HEADERS
    )


async def non_stream_chat_completion(message: str, model: str, get_mcp_client_func, get_system_message_func, llm_streaming) -> Dict[str, Any]:
    """Return single (non-streaming) chat completion response - collects streaming chunks into one response"""
    # Collect all streaming content and return as single response
    collected_content = ""
    async for chunk_data in _generate_streaming_response(message, model, get_mcp_client_func, get_system_message_func, llm_streaming):
        if chunk_data.startswith("data: ") and not chunk_data.startswith("data: [DONE]"):
            try:
                chunk_json = json.loads(chunk_data[6:])  # Remove "data: " prefix
                if "choices" in chunk_json and chunk_json["choices"]:
                    delta = chunk_json["choices"][0].get("delta", {})
                    if "content" in delta:
                        collected_content += delta["content"]
            except json.JSONDecodeError:
                continue
    
    # Return OpenAI-compatible single response format
    return {
        "id": f"chatcmpl-{hash(message) & 0x7fffffff:08x}",
        "object": "chat.completion",
        "created": int(asyncio.get_event_loop().time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": collected_content.strip()
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(message.split()),
            "completion_tokens": len(collected_content.split()),
            "total_tokens": len(message.split()) + len(collected_content.split())
        }
    }

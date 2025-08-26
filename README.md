
# AgentMCP Examples

A template for Agentic AI apps using LangChain (orchestration) + OpenRouter (LLMs) + FastMCP (MCP tools) + OpenWebUI (front end). The agent in this example is a LangChain ReAct agent.

## 🏗️ Pre-requisites

OpenAI-compatible API endpoint and API key (supports OpenAI, OpenRouter, and other compatible providers).

## 🏗️ Architecture

```
agentmcp_examples/
├── compose.yaml                 # Docker Compose orchestration
├── data_api/                    # Mock data provider (FastAPI)
├── mcp_servers/                 # MCP servers
│   ├── weather_server/          # Weather information MCP
│   ├── math_server/             # Mathematical calculations MCP
│   └── data_tool_server/        # Data analytics and insights MCP
└── agent_backends/
    └── react_simple/            # Main agent backend (LLM + Langchain + MCP)
```

## 🚀 Services

### 1. Data API (Port 9001)
- **Purpose**: Provides mock data for various use cases
- **Technology**: FastAPI
- **Endpoints**: `/users`, `/products`, `/documents`, `/search`

### 2. Agent Backend: react_simple (Port 9002)
- **Purpose**: Coordinates between LLM and MCP services
- **Technology**: FastAPI + Langchain + OpenAI-compatible providers
- **Location**: `agent_backends/react_simple/`
- **Features**: Tool detection, LLM integration, health monitoring, streaming responses
- **API Compatibility**: OpenAI-compatible endpoints for seamless integration

### 3. Weather MCP Server (Port 9200)
- **Purpose**: Handles weather-related queries
- **Technology**: FastMCP
- **Transport**: SSE (Server-Sent Events)

### 4. Math MCP Server (Port 9201)
- **Purpose**: Performs mathematical calculations
- **Technology**: FastMCP
- **Operations**: Addition, multiplication

### 5. Data Tool MCP Server (Port 9202)
- **Purpose**: Provides data analytics and insights
- **Technology**: FastMCP
- **Features**: User statistics, product analytics, document insights, data health checks
- **Transport**: SSE (Server-Sent Events)

### 6. OpenWebUI (Port 3000)
- **Purpose**: Modern chat interface with advanced features
- **Technology**: OpenWebUI (ChatGPT-like web interface)
- **Features**: 
  - Real-time streaming responses
  - Code interpreter with Python execution
  - Document upload and analysis
  - Model selection and configuration
  - Chat history and conversation management
  - Markdown and code syntax highlighting

## 🛠️ Setup Instructions

### Prerequisites
- Docker and Docker Compose
- API keys for your chosen providers (OpenRouter, NVIDIA, Cerebras, etc.)

### 1. Clone Repository
```bash
cd agentmcp-examples
```

### 2. Add API Keys (REQUIRED)

The default is using OPENROUTER_API_KEY.  To use other OpenAI compatible endpoints please refer to  [Customizing the LLM](#customizing-the-llm)

```bash
# Set your API key as an environment variable
export OPENROUTER_API_KEY="sk-or-v1-your-openrouter-key"
```



This environment variable is passed into the agent-backend service (see `compose.react_simple.yaml`).

To also add additional selectable endpoints, models, and keys, see the `agent_backends/react_simple/config.yaml` file.

docker compose up -d 

### 3. Deploy App

```bash
docker compose -f compose.react_simple.yaml up -d
```

# Test agent backend health
curl http://localhost:9002/health

# Test OpenAI-compatible models endpoint (should show your configured model)
curl http://localhost:9002/v1/models

# Expected response: {"object":"list","data":[{"id":"anthropic/claude-3.5-sonnet",...}]}

# Verify environment variables are set correctly
echo "Model configured: $LLM_MODEL"
echo "API base URL: $LLM_BASE_URL"  
echo "API key configured: ${OPENAI_API_KEY:+YES}"
```

### 5. Access Services
- **OpenWebUI**: http://localhost:3000 (Main chat interface)
- **Agent Backend (react_simple)**: http://localhost:9002
- **Data API**: http://localhost:9001
- **Weather MCP**: http://localhost:9200
- **Math MCP**: http://localhost:9201
- **Data Tool MCP**: http://localhost:9202

## 🧪 Testing the System

### Via OpenWebUI (Recommended)
1. Open http://localhost:3000
2. The interface will load with your configured model
3. Try example queries in the chat:
   - "What's the weather in Tokyo?"
   - "Calculate 25 * 8"
   - "What is 100 + 50?"
   - "Tell me about the users in the system"
   - "What products do we have?"
   - "Show me document statistics"
   - "Create a bar chart showing product sales data" (uses code interpreter)

### Via API (OpenAI-Compatible)
```bash
# Test streaming chat completions with your configured model
curl -X POST http://localhost:9002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d "{
    \"model\": \"$LLM_MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"What is 15 + 27?\"}],
    \"stream\": true
  }"

# Test non-streaming responses  
curl -X POST http://localhost:9002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d "{
    \"model\": \"$LLM_MODEL\", 
    \"messages\": [{\"role\": \"user\", \"content\": \"What is the weather in New York?\"}],
    \"stream\": false
  }"

# Test models endpoint (used by OpenWebUI)
curl http://localhost:9002/v1/models

# Test API key validation (should return warning message)
curl -X POST http://localhost:9002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$LLM_MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}],
    \"stream\": true
  }"

# Test health checks
curl http://localhost:9002/health
curl http://localhost:9001/health

# Test data API endpoints
curl http://localhost:9001/users
curl http://localhost:9001/products
curl http://localhost:9001/documents
curl http://localhost:9001/search?q=technology
```

## �🔧 Development

### Adding New MCP Servers
1. Create new directory in `mcp_servers/`
2. Add Dockerfile and requirements.txt
3. Implement MCP server using FastMCP
4. Update `compose.react_simple.yaml` to include new service
5. Modify `agent_backends/react_simple/main.py` to integrate new tools

#### Example: Data Tool MCP Server
The Data Tool MCP Server provides a template for creating new MCP servers that interact with data APIs:

```python
# Initialize FastMCP server
mcp = FastMCP("Data Tool Server", host="0.0.0.0", port=8080)

# Define a helper function for API calls
async def call_external_api(endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient() as client:
            url = f"{API_BASE_URL}{endpoint}"
            response = await client.get(url, params=params or {})
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Error calling API {endpoint}: {e}")
        return {"error": str(e)}

# Define a tool with required parameters
@mcp.tool()
async def my_tool(query_type: str) -> ToolResult:
    """
    Tool description here.
    
    Args:
        query_type: Type of query to perform (e.g., "all", "summary", "details")
        
    Returns:
        Tool results description
    """
    try:
        # Call external API
        data = await call_external_api("/endpoint")
        
        # Process data
        result = process_data(data)
        
        # Return structured result
        return ToolResult(structured_content=result)
    except Exception as e:
        return ToolResult(structured_content={"error": str(e)})
```

**Important Tips:**
- Always use required and typed parameters instead of optional parameters with default values. This because the MCP arg_schema validation will be based off these definitions.
- Use descriptive parameter names (e.g., `query_type`, `check_type`)
- Return results using `ToolResult` with structured content
- Handle API responses that might be lists or dictionaries
- Include proper error handling

### Customizing the LLM


To add or change LLMs available to the system, update the `agent_backends/react_simple/config.yaml` file. Each entry in the `models` section defines a model, its provider, and the environment variable used for its API key. For example:

```yaml
models:
  - id: "anthropic/claude-3.5-sonnet"
    provider: "openrouter"
    api_key_env: "OPENROUTER_API_KEY"
  - id: "openai/gpt-4o"
    provider: "openai"
    api_key_env: "OPENAI_API_KEY"
```


For each model you add, ensure the corresponding API key environment variable is set in your shell, and the `compose.yaml` under the `environment:` section for the `agent-backend` service:

```yaml
services:
  agent-backend:
    environment:
      - OPENROUTER_API_KEY=sk-or-v1-your-openrouter-key

      - OPENAI_API_KEY=sk-your-openai-key
```

**Important:** All models listed in `config.yaml` must support tool calling (function calling) for the system to work correctly. If a model does not support tool calls, it will not be able to use the MCP tools and may cause errors.


docker compose restart agent-backend

After updating `config.yaml` and environment variables, restart the agent-backend service:

```bash
docker compose -f compose.react_simple.yaml restart agent-backend
```

You can now select the newly available models in OpenWebUI or via the API.


## 🧠 LLM Tool Integration


## 📊 Validation

### Health Checks and Logs

- **Orchestration API**: `GET /health`
- **Data API**: `GET /health`
- **Individual Services**: Check via Docker logs

The Data API provides a comprehensive health check endpoint that returns:
- Service status
- API version
- Data counts (users, products, documents)
- Timestamp information

To view logs:
```bash
docker compose logs
docker compose logs agent-backend
# View all logs
docker compose -f compose.react_simple.yaml logs

# View specific service logs
docker compose -f compose.react_simple.yaml logs agent-backend
```

## 🤝 Contributing

This template is designed for prototyping Agent apps. Feel free to:
- Add new MCP servers and tools
- Enhance the UI
- Add new data apis
- Implement additional LLM providers

## 🐛 Troubleshooting

### Common Issues

**Services not starting:**
docker compose down
docker compose build --no-cache
docker compose up -d
```bash
docker compose -f compose.react_simple.yaml down
docker compose -f compose.react_simple.yaml build --no-cache
docker compose -f compose.react_simple.yaml up -d
```

**MCP servers not responding:**
- Check if ports are available
- Verify MCP server logs: `docker compose -f compose.react_simple.yaml logs weather-mcp`
- For Data Tool MCP: `docker compose -f compose.react_simple.yaml logs data-tool-mcp`

## 📚 Additional Resources

- [OpenWebUI Documentation](https://docs.openwebui.com/)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [Langchain Documentation](https://python.langchain.com/)
- [OpenRouter API](https://openrouter.ai/)
- [Docker Compose Reference](https://docs.docker.com/compose/)

---

**Happy Coding with AgentMCP! 🎉**
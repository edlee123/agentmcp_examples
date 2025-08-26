from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

app = FastAPI(title="Mock Data API", description="Provides mock data for AI applications")

# Sample data structures
class User(BaseModel):
    id: int
    name: str
    email: str
    created_at: str

class Product(BaseModel):
    id: int
    name: str
    price: float
    category: str
    description: str

class Document(BaseModel):
    id: int
    title: str
    content: str
    tags: List[str]
    created_at: str

# Mock data
users_data = [
    {"id": 1, "name": "Alice Johnson", "email": "alice@example.com", "created_at": "2024-01-15T10:30:00Z"},
    {"id": 2, "name": "Bob Smith", "email": "bob@example.com", "created_at": "2024-01-16T14:20:00Z"},
    {"id": 3, "name": "Carol Davis", "email": "carol@example.com", "created_at": "2024-01-17T09:15:00Z"},
]

products_data = [
    {"id": 1, "name": "Laptop Pro", "price": 1299.99, "category": "Electronics", "description": "High-performance laptop for professionals"},
    {"id": 2, "name": "Wireless Headphones", "price": 199.99, "category": "Electronics", "description": "Premium noise-canceling headphones"},
    {"id": 3, "name": "Coffee Maker", "price": 89.99, "category": "Appliances", "description": "Programmable coffee maker with timer"},
]

documents_data = [
    {"id": 1, "title": "AI in Healthcare", "content": "Artificial Intelligence is revolutionizing healthcare through improved diagnostics, personalized treatment plans, and drug discovery.", "tags": ["AI", "Healthcare", "Technology"], "created_at": "2024-01-10T08:00:00Z"},
    {"id": 2, "title": "Climate Change Solutions", "content": "Renewable energy sources like solar and wind power are key to addressing climate change and reducing carbon emissions.", "tags": ["Climate", "Environment", "Energy"], "created_at": "2024-01-11T12:30:00Z"},
    {"id": 3, "title": "Future of Work", "content": "Remote work and automation are reshaping the modern workplace, requiring new skills and adaptability from workers.", "tags": ["Work", "Technology", "Future"], "created_at": "2024-01-12T16:45:00Z"},
]

@app.get("/")
async def root():
    return {
        "message": "Mock Data API is running",
        "endpoints": ["/users", "/products", "/documents", "/search", "/search/docs", "/health"],
        "description": "FastAPI-based mock data provider for AI applications",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint that returns the status of the API and its components.
    Used for monitoring and service discovery.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "data_counts": {
            "users": len(users_data),
            "products": len(products_data),
            "documents": len(documents_data)
        },
        "uptime": "unknown"  # In a real app, you would track the start time and calculate uptime
    }

@app.get("/search/docs")
async def search_documentation():
    """
    Detailed documentation for the search functionality.
    """
    return {
        "endpoint": "/search",
        "description": "Universal search across all data types with flexible filtering",
        "parameters": {
            "q": {
                "type": "string",
                "required": True,
                "description": "Search query string (case-insensitive)"
            },
            "type": {
                "type": "string",
                "required": False,
                "options": ["users", "products", "documents"],
                "description": "Filter search to specific data type. If omitted, searches all types."
            }
        },
        "search_fields": {
            "users": ["name", "email"],
            "products": ["name", "description", "category"],
            "documents": ["title", "content", "tags[]"]
        },
        "algorithm": {
            "type": "substring_matching",
            "case_sensitive": False,
            "partial_matches": True,
            "description": "Simple text search using Python 'in' operator on lowercase strings"
        },
        "examples": [
            {
                "url": "/search?q=alice",
                "description": "Search for 'alice' across all data types"
            },
            {
                "url": "/search?q=laptop&type=products",
                "description": "Search for 'laptop' only in products"
            },
            {
                "url": "/search?q=AI&type=documents",
                "description": "Search for 'AI' only in documents"
            },
            {
                "url": "/search?q=technology",
                "description": "Search for 'technology' across all types (will find in document tags)"
            }
        ],
        "response_format": {
            "query": "original search term",
            "results": [
                {
                    "type": "data_type_name",
                    "data": "complete_data_object"
                }
            ],
            "count": "number_of_results",
            "search_info": {
                "query_processed": "lowercase_version_of_query",
                "data_types_searched": ["list_of_searched_types"],
                "algorithm": "search_algorithm_used"
            }
        }
    }

@app.get("/users", response_model=List[User])
async def get_users():
    return users_data

@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int):
    user = next((u for u in users_data if u["id"] == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.get("/products", response_model=List[Product])
async def get_products():
    return products_data

@app.get("/products/{product_id}", response_model=Product)
async def get_product(product_id: int):
    product = next((p for p in products_data if p["id"] == product_id), None)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product

@app.get("/documents", response_model=List[Document])
async def get_documents():
    return documents_data

@app.get("/documents/{doc_id}", response_model=Document)
async def get_document(doc_id: int):
    doc = next((d for d in documents_data if d["id"] == doc_id), None)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc

@app.get("/search")
async def search_data(q: str, type: Optional[str] = None):
    """
    Universal search endpoint that performs text-based search across all data types.
    
    **How it works:**
    1. Converts the search query to lowercase for case-insensitive matching
    2. Searches across different data types based on the 'type' parameter:
       - If no type specified: searches ALL data types (users, products, documents)
       - If type specified: searches only that specific data type
    
    **Search Fields by Data Type:**
    - **Users**: Searches in 'name' and 'email' fields
    - **Products**: Searches in 'name', 'description', and 'category' fields
    - **Documents**: Searches in 'title', 'content', and 'tags' array
    
    **Search Algorithm:**
    - Uses simple substring matching (case-insensitive)
    - For documents: also searches within individual tags using any() function
    - Returns partial matches (e.g., "lap" will match "Laptop Pro")
    
    **Parameters:**
    - q (required): The search query string
    - type (optional): Filter by data type ("users", "products", "documents")
    
    **Response Format:**
    ```json
    {
        "query": "original search term",
        "results": [
            {
                "type": "data_type",
                "data": { ... actual data object ... }
            }
        ],
        "count": number_of_results
    }
    ```
    
    **Example Usage:**
    - `/search?q=alice` - Search for "alice" across all data types
    - `/search?q=laptop&type=products` - Search for "laptop" only in products
    - `/search?q=AI&type=documents` - Search for "AI" only in documents
    """
    results = []
    query_lower = q.lower()
    
    # Search in Users data type
    if not type or type == "users":
        for user in users_data:
            # Search in name and email fields
            if (query_lower in user["name"].lower() or
                query_lower in user["email"].lower()):
                results.append({"type": "user", "data": user})
    
    # Search in Products data type
    if not type or type == "products":
        for product in products_data:
            # Search in name, description, and category fields
            if (query_lower in product["name"].lower() or
                query_lower in product["description"].lower() or
                query_lower in product["category"].lower()):
                results.append({"type": "product", "data": product})
    
    # Search in Documents data type
    if not type or type == "documents":
        for doc in documents_data:
            # Search in title, content, and tags array
            if (query_lower in doc["title"].lower() or
                query_lower in doc["content"].lower() or
                any(query_lower in tag.lower() for tag in doc["tags"])):
                results.append({"type": "document", "data": doc})
    
    return {
        "query": q,
        "results": results,
        "count": len(results),
        "search_info": {
            "query_processed": query_lower,
            "data_types_searched": ["users", "products", "documents"] if not type else [type],
            "algorithm": "case-insensitive substring matching"
        }
    }

@app.post("/users", response_model=User)
async def create_user(user: User):
    """Create a new user (for demo purposes)"""
    users_data.append(user.dict())
    return user

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
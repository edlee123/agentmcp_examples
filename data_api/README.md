
# Mock Data API Documentation

## 🧩 Overview

This `data_api` is a **mock API** designed for development, testing, and demonstration purposes. It simulates a typical data service with endpoints for searching and retrieving user, product, and document data. The structure and logic are intended to be easily extensible—users can define their own mock APIs by modifying or extending the provided code and data.

The API is ideal for:
- Prototyping frontend or backend integrations
- Interview exercises and technical assessments
- Testing API clients without a real backend
- Experimenting with search and filtering logic

**Note:** All data is in-memory and non-persistent. You are encouraged to adapt the data models, endpoints, and logic to fit your own mock API needs.


## 🔍 Search Endpoint

The `/search` endpoint provides a flexible search capability across all data types in the mock API. It uses **case-insensitive substring matching** to find relevant results across users, products, and documents.


## 🎯 How Search Works

### Search Algorithm
1. **Input Processing**: Converts search query to lowercase for case-insensitive matching
2. **Type Filtering**: Searches a specific data type if the `type` parameter is provided, otherwise searches all types
3. **Field Matching**: Uses Python's `in` operator for substring matching across relevant fields
4. **Result Compilation**: Returns all matches with complete data objects and metadata

### Search Strategy
- **Partial Matching**: e.g., "lap" matches "Laptop Pro"
- **Case Insensitive**: e.g., "ALICE" matches "alice johnson"
- **Multi-field**: Searches across multiple fields per data type
- **Cross-type**: Can search all data types simultaneously


## 📋 Searchable Fields by Data Type

| Data Type   | Searchable Fields                  | Example Query                                 |
|-------------|------------------------------------|------------------------------------------------|
| **users**   | `name`, `email`                    | `?q=alice` → finds "Alice Johnson"            |
| **products**| `name`, `description`, `category`  | `?q=laptop` → finds "Laptop Pro"              |
| **documents**| `title`, `content`, `tags[]`      | `?q=AI` → finds AI-related documents           |


## 🔧 API Reference

### Endpoint
```
GET /search
```

### Parameters
| Parameter | Type   | Required | Description                                             |
|-----------|--------|----------|---------------------------------------------------------|
| `q`       | string | ✅ Yes   | Search query string (case-insensitive)                  |
| `type`    | string | ❌ No    | Filter by data type: "users", "products", "documents" |

### Response Format
```json
{
  "query": "original search term",
  "results": [
    {
      "type": "user|product|document",
      "data": { /* complete data object */ }
    }
  ],
  "count": 2,
  "search_info": {
    "query_processed": "lowercase version of query",
    "data_types_searched": ["users", "products", "documents"],
    "algorithm": "case-insensitive substring matching"
  }
}
```


## 📚 Example Queries

### 1. Universal Search (All Data Types)
```bash
GET /search?q=alice
```
**What it searches:**
- Users: name, email fields
- Products: name, description, category fields  
- Documents: title, content, tags fields

**Expected Results:**
- Finds "Alice Johnson" in users
- Case-insensitive: also matches "ALICE" or "Alice"

### 2. Type-Specific Search
```bash
GET /search?q=laptop&type=products
```
**What it searches:**
- Only products data
- Searches: name, description, category

**Expected Results:**
- Finds "Laptop Pro" product
- Ignores users and documents entirely

### 3. Content-Based Search
```bash
GET /search?q=AI&type=documents
```
**What it searches:**
- Only documents data
- Searches: title, content, tags array

**Expected Results:**
- Finds "AI in Healthcare" document
- Matches in title, content, or tags

### 4. Tag-Based Search
```bash
GET /search?q=technology
```
**What it searches:**
- All data types
- For documents: includes tags array search

**Expected Results:**
- Finds documents with "Technology" in tags
- Also finds any content with "technology" text

### 5. Email Domain Search
```bash
GET /search?q=example.com&type=users
```
**What it searches:**
- Only users data
- Searches: name, email fields

**Expected Results:**
- Finds all users with "@example.com" emails
- Demonstrates partial matching in email field

### 6. Category Search
```bash
GET /search?q=electronics&type=products
```
**What it searches:**
- Only products data
- Searches: name, description, category

**Expected Results:**
- Finds products in "Electronics" category
- Case-insensitive: matches "electronics", "Electronics", "ELECTRONICS"

## 🧪 Testing Examples

### Using curl:
```bash
# Basic search across all types
curl "http://localhost:8001/search?q=alice"

# Type-specific search
curl "http://localhost:8001/search?q=laptop&type=products"

# Case insensitivity test
curl "http://localhost:8001/search?q=TECHNOLOGY"

# Partial matching test
curl "http://localhost:8001/search?q=lap"

# Email domain search
curl "http://localhost:8001/search?q=example.com&type=users"

# Empty results test
curl "http://localhost:8001/search?q=nonexistent"
```

### Using Python:
```python
import httpx

async def test_search():
    async with httpx.AsyncClient() as client:
        # Search across all types
        response = await client.get("http://localhost:8001/search?q=alice")
        print(response.json())
        
        # Type-specific search
        response = await client.get("http://localhost:8001/search?q=laptop&type=products")
        print(response.json())
```


## 📊 Sample Data Overview

The default mock data includes a small set of users, products, and documents for demonstration. You can modify or extend these records in the code or data files to suit your own mock API scenarios.

### Users (3 records)
- Alice Johnson (alice@example.com)
- Bob Smith (bob@example.com)
- Carol Davis (carol@example.com)

### Products (3 records)
- Laptop Pro (Electronics, $1299.99)
- Wireless Headphones (Electronics, $199.99)
- Coffee Maker (Appliances, $89.99)

### Documents (3 records)
- "AI in Healthcare" (tags: AI, Healthcare, Technology)
- "Climate Change Solutions" (tags: Climate, Environment, Energy)
- "Future of Work" (tags: Work, Technology, Future)


## 🔍 Search Behavior Examples

| Query | Type Filter | Matches Found | Explanation |
|-------|-------------|---------------|-------------|
| `alice` | none | 1 user | Matches "Alice Johnson" in name |
| `ALICE` | none | 1 user | Case-insensitive matching |
| `example.com` | users | 3 users | All users have @example.com emails |
| `laptop` | products | 1 product | Matches "Laptop Pro" |
| `electronics` | products | 2 products | Matches category field |
| `AI` | documents | 1 document | Matches title and tags |
| `technology` | none | 3 documents | Matches tags in multiple docs |
| `nonexistent` | none | 0 results | No matches found |


## 🚀 Advanced Usage & Extensibility

### Customizing the Mock API
- **Add or modify endpoints**: Extend the API with new routes or logic as needed
- **Change data models**: Adapt the user, product, or document schemas
- **Swap in your own data**: Replace or expand the in-memory data for different scenarios

### Search Documentation Endpoint
```bash
GET /search/docs
```
Returns comprehensive API documentation for the search functionality.

### Performance Considerations
- **In-Memory Search**: Fast for small datasets (current implementation)
- **Linear Scan**: O(n) complexity per data type
- **String Operations**: Uses Python's optimized string `in` operator
- **Scalability**: For larger datasets, consider database indexing or external search engines

### Potential Enhancements
- **Fuzzy Matching**: Levenshtein distance for typo tolerance
- **Ranking**: Score results by relevance
- **Highlighting**: Mark matched terms in results
- **Pagination**: Limit results for large datasets
- **Filters**: Additional filtering by date, price, etc.
- **Full-Text Search**: Integration with Elasticsearch or similar


## 🔧 Implementation Details

The search function iterates through each data type and performs substring matching:

```python
# Simplified algorithm
def search_data(query: str, type_filter: str = None):
    results = []
    query_lower = query.lower()
    # Search users
    if not type_filter or type_filter == "users":
        for user in users_data:
            if (query_lower in user["name"].lower() or 
                query_lower in user["email"].lower()):
                results.append({"type": "user", "data": user})
    # Similar logic for products and documents...
    return results
```

This provides a simple but effective search capability suitable for development, testing, and interview scenarios. You can adapt this logic for your own mock API needs.
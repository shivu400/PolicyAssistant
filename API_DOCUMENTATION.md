# API Documentation for Policy Assistant Webhook

## Endpoint

```
POST /hackrx/run
```

## Authentication

The API requires authentication using a Bearer token in the Authorization header:

```
Authorization: Bearer c96a70a0e0ca2611043c5b543b5b6f5940d3bf5c480c5ac1236690b2f148783c
```

## Request Format

The request must be a JSON payload with the following structure:

```json
{
  "documents": "https://example.com/path/to/document.pdf",
  "questions": ["What is covered under this policy?", "What are the exclusions?"]
}
```

### Parameters

- `documents`: A URL pointing to the policy document (PDF, DOCX, or EML format)
- `questions`: An array of questions to ask about the policy document

## Response Format

The API returns a JSON response with the following structure:

```json
{
  "answers": ["Answer to question 1", "Answer to question 2"]
}
```

### Response Fields

- `answers`: An array of answers corresponding to each question in the request

## Error Responses

- **401 Unauthorized**: Invalid or missing authentication token
- **400 Bad Request**: Invalid request format or missing required fields
- **500 Internal Server Error**: Error downloading or processing the document

## Example Usage

### cURL

```bash
curl -X POST \
  https://your-render-url.onrender.com/hackrx/run \
  -H 'Authorization: Bearer c96a70a0e0ca2611043c5b543b5b6f5940d3bf5c480c5ac1236690b2f148783c' \
  -H 'Content-Type: application/json' \
  -d '{
    "documents": "https://example.com/path/to/policy.pdf",
    "questions": ["What is the coverage for hospitalization?", "What are the exclusions for pre-existing conditions?"]
  }'
```

### Python

```python
import requests
import json

url = "https://your-render-url.onrender.com/hackrx/run"

headers = {
    "Authorization": "Bearer c96a70a0e0ca2611043c5b543b5b6f5940d3bf5c480c5ac1236690b2f148783c",
    "Content-Type": "application/json"
}

payload = {
    "documents": "https://example.com/path/to/policy.pdf",
    "questions": ["What is the coverage for hospitalization?", "What are the exclusions for pre-existing conditions?"]
}

response = requests.post(url, headers=headers, data=json.dumps(payload))

print(response.status_code)
print(response.json())
```
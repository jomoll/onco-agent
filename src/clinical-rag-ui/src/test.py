import requests
import json

# Test the API
BASE_URL = "http://localhost:5000/api"

def test_patient_search():
    print("Testing patient search...")
    response = requests.post(f"{BASE_URL}/patients/search", 
                           json={"query": "jane"})
    print(f"Status: {response.status_code}")
    print(f"Results: {response.json()}")

def test_document_search():
    print("\nTesting document search...")
    response = requests.post(f"{BASE_URL}/documents/search", 
                           json={
                               "query": "blood pressure",
                               "patient_id": "p001",
                               "k": 3
                           })
    print(f"Status: {response.status_code}")
    results = response.json()
    print(f"Found {len(results)} documents")
    for doc in results:
        print(f"  {doc['title']} (Score: {doc['score']:.2f})")
        print(f"    {doc['snippet'][:100]}...")

if __name__ == "__main__":
    try:
        test_patient_search()
        test_document_search()
    except requests.exceptions.ConnectionError:
        print("Error: API server not running. Start with: python api_server.py")
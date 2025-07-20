import requests
import json

# API base URL (adjust if running on different host/port)
BASE_URL = "http://localhost:8000"

def test_work_assistant():
    """Test the work assistant API endpoints"""
    print("=== Testing Work Assistant ===")
    
    # Create work tasks
    work_task_data = {
        "message": "Create or update few ToDos: 1) Re-film Module 6, lesson 5 by end of day today. 2) Update audioUX by next Monday.",
        "user_id": "test_user"
    }
    
    response = requests.post(f"{BASE_URL}/work/tasks", json=work_task_data)
    print(f"Create Work Tasks Response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()
    
    # Get all work tasks
    response = requests.get(f"{BASE_URL}/work/tasks")
    print(f"Get Work Tasks Response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()
    
    # Get work summary
    summary_data = {"message": "Give me a todo summary"}
    response = requests.post(f"{BASE_URL}/work/summary", json=summary_data)
    print(f"Work Summary Response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_personal_assistant():
    """Test the personal assistant API endpoints"""
    print("=== Testing Personal Assistant ===")
    
    # Create personal tasks
    personal_task_data = {
        "message": "Create ToDos: 1) Check on swim lessons for the baby this weekend. 2) For winter travel, check AmEx points.",
        "user_id": "test_user"
    }
    
    response = requests.post(f"{BASE_URL}/personal/tasks", json=personal_task_data)
    print(f"Create Personal Tasks Response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()
    
    # Get all personal tasks
    response = requests.get(f"{BASE_URL}/personal/tasks")
    print(f"Get Personal Tasks Response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()
    
    # Get personal summary
    summary_data = {"message": "Give me a todo summary"}
    response = requests.post(f"{BASE_URL}/personal/summary", json=summary_data)
    print(f"Personal Summary Response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_health_check():
    """Test health check endpoint"""
    print("=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health Check Response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

if __name__ == "__main__":
    try:
        # Test health check first
        test_health_check()
        
        # Test work assistant
        test_work_assistant()
        
        # Test personal assistant
        test_personal_assistant()
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        print("Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"Error: {e}")


# Alternative: Using curl commands
curl_examples = """
# Test with curl commands:

# Health check
curl -X GET "http://localhost:8000/health"

# Create work tasks
curl -X POST "http://localhost:8000/work/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create or update few ToDos: 1) Re-film Module 6, lesson 5 by end of day today. 2) Update audioUX by next Monday.",
    "user_id": "test_user"
  }'

# Get all work tasks
curl -X GET "http://localhost:8000/work/tasks"

# Get work summary
curl -X POST "http://localhost:8000/work/summary" \
  -H "Content-Type: application/json" \
  -d '{"message": "Give me a todo summary"}'

# Create personal tasks
curl -X POST "http://localhost:8000/personal/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create ToDos: 1) Check on swim lessons for the baby this weekend. 2) For winter travel, check AmEx points.",
    "user_id": "test_user"
  }'

# Get all personal tasks
curl -X GET "http://localhost:8000/personal/tasks"

# Complete a task (replace TASK_ID with actual task ID)
curl -X PUT "http://localhost:8000/work/tasks/TASK_ID/complete"
"""

print(curl_examples)
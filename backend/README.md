Langgraph assistant backend

This is a backend for langgraph assistant.

To run the backend, you need to have python installed.

# Install dependencies
pip install -r requirements.txt

# Run the backend
python task_api.py 
# Alternative
uvicorn task_api:app --reload
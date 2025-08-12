Langgraph assistant backend

This is a backend for langgraph assistant.

To run the backend, you need to have python installed.

# Install dependencies
pip install -r requirements.txt

# Run the backend
python task_api.py 
# Alternative
uvicorn task_api:app --reload
# For https
uvicorn task_api:app --reload --port 8000 --ssl-keyfile localhost.key --ssl-certfile localhost.crt

# Task assistants
task_assistant : langchain
task_assistant2 : langgraph + qdrant
task_assistant3 : langgraph + pgvector

### command to generate self signed certificate for localhost
openssl req -x509 -out localhost.crt -keyout localhost.key \
  -newkey rsa:2048 -nodes -sha256 \
  -subj '/CN=localhost' -extensions EXT -config <( \
   printf "[dn]\nCN=localhost\n[req]\ndistinguished_name = dn\n[EXT]\nsubjectAltName=DNS:localhost\nkeyUsage=digitalSignature\nextendedKeyUsage=serverAuth")

### for Google chrome only
#### import the certificate that was generated
chrome://certificate-manager/localcerts
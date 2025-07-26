from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.documents import Document
#from langchain_community.vectorstores import InMemoryVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime, timedelta
from typing import Dict, List, Any, TypedDict, Annotated, Optional
import json
import uuid
import os
from dotenv import load_dotenv
import operator
import re

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["QDRANT_API_KEY"] = os.getenv("QDRANT_API_KEY")

# State definition for LangGraph
class TaskState2(TypedDict):
    messages: Annotated[List[dict], operator.add]
    user_input: str
    tasks: List[Dict[str, Any]]
    created_tasks: List[Dict[str, Any]]
    updated_tasks: List[Dict[str, Any]]
    intent: str  # create, update, complete, summary, query
    category: str  # work, personal
    user_id: str
    response: str
    extracted_info: Dict[str, Any]

# NOTE: Langgraph task assistant
class TaskAssistant2:
    def __init__(self, role_prompt: str, category: str, user_id: str, qdrant_url: str):
        self.role_prompt = role_prompt
        self.category = category
        self.user_id = user_id
        self.qdrant_url = qdrant_url

        # LLM setup
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            self.llm = ChatGroq(temperature=0.7, model_name="llama-3.3-70b-versatile")
        else:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                self.llm = ChatOpenAI(model="gpt-4")
            else:
                raise ValueError("Neither GROQ_API_KEY nor OPENAI_API_KEY is set")

        # Task tracking
        self.tasks = []  # In-memory storage
        self.conversation_history = []
        self.task_id_to_doc_id = {}  # Map task IDs to document IDs in vector store
        
        # Setup embeddings for vector store
        self.embeddings = self._setup_embeddings()
        
        # Initialize InMemoryVectorStore for task persistence
        # self.vector_store = InMemoryVectorStore(embedding=self.embeddings)
        
        # Initialize Qdrant client and vector store
        self.qdrant_client = QdrantClient(
            url=self.qdrant_url,
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        self.collection_name = f"tasks_{category}_{user_id}"
        self._setup_qdrant_collection()
        
        # ensure index is present
        self._ensure_task_id_index()

        # Initialize Qdrant vector store
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embeddings
        )
        # Create LangGraph workflow
        self.workflow = self._create_workflow()

        
    def _setup_embeddings(self):
        """Setup embeddings for vector store"""
        # Use a lightweight model
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    def _setup_qdrant_collection(self):
        """Setup Qdrant collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                # Create collection with vector configuration
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,  # all-MiniLM-L6-v2 produces 384-dimensional vectors
                        distance=Distance.COSINE
                    )
                )
                print(f"Created Qdrant collection: {self.collection_name}")
            else:
                print(f"Using existing Qdrant collection: {self.collection_name}")
                
        except Exception as e:
            print(f"Error setting up Qdrant collection: {e}")
            raise

    def _task_to_document(self, task: Dict[str, Any]) -> Document:
        """Convert a task dictionary to a Document for vector storage"""
        # Create searchable content from task
        content_parts = [
            f"Title: {task.get('title', '')}",
            f"Description: {task.get('description', '')}",
            f"Category: {task.get('category', '')}",
            f"Status: {task.get('status', 'pending')}",
            f"Priority: {task.get('priority', 'medium')}",
        ]
        
        if task.get('tags'):
            content_parts.append(f"Tags: {', '.join(task['tags'])}")
        
        if task.get('deadline'):
            content_parts.append(f"Deadline: {task['deadline']}")
        
        content = "\n".join(content_parts)
        
        # Store full task data in metadata
        metadata = task.copy()
        metadata['searchable_content'] = content
        
        return Document(
            page_content=content,
            metadata=metadata
        )
    
    def _add_task_to_vector_store(self, task: Dict[str, Any]) -> str:
        """Add a task to the vector store and return the document ID"""
        document = self._task_to_document(task)
        doc_ids = self.vector_store.add_documents([document])
        doc_id = doc_ids[0]
        print(f"Added task to vector store: {doc_id}")

        # Map task ID to document ID
        self.task_id_to_doc_id[task['id']] = doc_id
        
        return doc_id
    
    def _update_task_in_vector_store(self, task: Dict[str, Any]):
        """Update a task in the vector store"""
        task_id = task['id']
        if task_id in self.task_id_to_doc_id:
            # Remove old document
            doc_id = self.task_id_to_doc_id[task_id]
            try:
                self.vector_store.delete([doc_id])
            except:
                pass  # Document might not exist
        
        # Add updated document
        self._add_task_to_vector_store(task)
    
    def _search_tasks(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search tasks using vector similarity"""
        try:
            results = self.vector_store.similarity_search(query, k=k)
            tasks = []
            for doc in results:
                if 'id' in doc.metadata:
                    tasks.append(doc.metadata)
            return tasks
        except Exception as e:
            print(f"Search error: {e}")
            return self.tasks  # Fallback to all tasks
    
    def _get_all_tasks_from_vector_store(self) -> List[Dict[str, Any]]:
        """Get all tasks from vector store"""
        try:
            # Try multiple search strategies
            all_results = []
            all_results = self.vector_store.similarity_search("task todo work personal", k=1000)
        
            # Strategy 1: Empty query (gets everything in some implementations)
            #results1 = self.vector_store.similarity_search("", k=1000)
            #all_results.extend(results1)
        
            # Strategy 2: Very broad terms
            #results2 = self.vector_store.similarity_search("task", k=1000)
            #all_results.extend(results2)
        
            # Strategy 3: If your vector store supports it, get all documents
            # results3 = self.vector_store.get_all_documents()  # Implementation dependent
        
            print(f'Found {len(all_results)} total results')
        
            tasks = []
            seen_ids = set()
        
            for doc in all_results:
                if 'id' in doc.metadata and doc.metadata['id'] not in seen_ids:
                    # Filter by category and user
                    if (doc.metadata.get('category') == self.category and 
                        doc.metadata.get('user_id') == self.user_id):
                        print(f"Found task {doc.metadata}.")
                        tasks.append(doc.metadata)
                        seen_ids.add(doc.metadata['id'])
            
            return tasks
        except Exception:
            return self.tasks  # Fallback to in-memory tasks
        
    def _create_workflow(self):
        """Create the LangGraph workflow for task processing"""
        # Create the state graph
        workflow = StateGraph(TaskState2)
        
        # Add nodes
        workflow.add_node("analyze_intent", self._analyze_intent)
        workflow.add_node("extract_task_info", self._extract_task_info)
        workflow.add_node("create_tasks", self._create_tasks)
        workflow.add_node("update_tasks", self._update_tasks)
        workflow.add_node("complete_tasks", self._complete_tasks)
        workflow.add_node("generate_summary", self._generate_summary)
        workflow.add_node("generate_response", self._generate_response)
        
        # Set entry point
        workflow.set_entry_point("analyze_intent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "analyze_intent",
            self._route_by_intent,
            {
                "create": "extract_task_info",
                "update": "extract_task_info", 
                "complete": "complete_tasks",
                "summary": "generate_summary",
                "query": "generate_response"
            }
        )
        
        # Add edges from extract_task_info to appropriate actions
        workflow.add_conditional_edges(
            "extract_task_info",
            self._route_by_action,
            {
                "create": "create_tasks",
                "update": "update_tasks"
            }
        )
        
        # All action nodes lead to response generation
        workflow.add_edge("create_tasks", "generate_response")
        workflow.add_edge("update_tasks", "generate_response")
        workflow.add_edge("complete_tasks", "generate_response")
        workflow.add_edge("generate_summary", "generate_response")
        
        # Response generation ends the workflow
        workflow.add_edge("generate_response", END)
        
        # Compile the workflow
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def _analyze_intent(self, state: TaskState2) -> TaskState2:
        """Analyze user intent from the input"""
        user_input = state["user_input"].lower()
        
        # Intent classification logic
        if any(keyword in user_input for keyword in ["create", "add", "new", "todo"]):
            intent = "create"
        elif any(keyword in user_input for keyword in ["update", "modify", "change", "edit"]):
            intent = "update"
        elif any(keyword in user_input for keyword in ["complete", "done", "finished", "mark"]):
            intent = "complete"
        elif any(keyword in user_input for keyword in ["summary", "list", "show", "overview"]):
            intent = "summary"
        else:
            intent = "query"
        
        state["intent"] = intent
        state["messages"].append({"role": "system", "content": f"Intent classified as: {intent}"})
        return state
    
    def _route_by_intent(self, state: TaskState2) -> str:
        """Route based on detected intent"""
        return state["intent"]
    
    def _route_by_action(self, state: TaskState2) -> str:
        """Route based on the action needed after info extraction"""
        return state["intent"]
    
    def _extract_task_info(self, state: TaskState2) -> TaskState2:
        """Extract task information using LLM"""
        print(f"Extracting task info: {state['user_input']}")
        extraction_prompt = """
        Extract task information from the user input. Return a JSON object with the following structure:
        {{
            "tasks": [
                {{
                    "title": "task title",
                    "description": "optional description",
                    "deadline": "ISO format datetime if mentioned",
                }}
            ]
        }}
        
        User input: {user_input}
        """
        print(extraction_prompt.format(user_input=state["user_input"]))

        try:
            print("Extracting task info...")
            response = self.llm.invoke([HumanMessage(content=extraction_prompt.format(user_input=state["user_input"]))])
            
            print(f"Response: {response}")
            json_match = re.findall(r'```json\n(.*?)\n```', response.content, re.DOTALL)
            print(f"JSON match: {json_match}")
            if len(json_match) == 0:
                raise Exception("No JSON match found")
            print(f"last match: {json_match[-1]}")
            if json_match:
                extracted_info = json.loads(json_match[-1])
            else:
                # Fallback to basic extraction
                extracted_info = self._basic_task_extraction(state["user_input"])
            
            state["extracted_info"] = extracted_info
            print(f"Extracted task info: {state['extracted_info']}")
        except Exception as e:
            print(f"Extraction failed: {str(e)}")
            # Fallback to basic extraction
            state["extracted_info"] = self._basic_task_extraction(state["user_input"])
            state["messages"].append({"role": "system", "content": f"Extraction fallback used: {str(e)}"})
        
        return state
    
    def _basic_task_extraction(self, user_input: str) -> Dict[str, Any]:
        """Fallback basic task extraction"""
        tasks = []
        lines = user_input.split('\n')
        
        for line in lines:
            if any(marker in line for marker in ['1)', '2)', '3)', '4)', '5)', '-', '*']):
                task_text = line.strip()
                # Remove numbering/bullets
                for marker in ['1)', '2)', '3)', '4)', '5)', '-', '*']:
                    task_text = task_text.replace(marker, '').strip()
                
                if task_text:
                    task = {"title": task_text}
                    deadline = self._extract_deadline(task_text)
                    if deadline:
                        task["deadline"] = deadline
                    tasks.append(task)
        
        return {"tasks": tasks}
    
    def _create_tasks(self, state: TaskState2) -> TaskState2:
        """Create new tasks based on extracted information"""
        created_tasks = []
        print(f"Creating tasks: {state['user_input']}")
        
        for task_info in state["extracted_info"].get("tasks", []):
            task = {
                'id': str(uuid.uuid4()),
                'title': task_info.get('title', ''),
                'description': task_info.get('description', ''),
                'status': 'pending',
                'created_at': datetime.now().isoformat(),
                'category': self.category,
                'user_id': self.user_id,
                'completed': False
            }
            
            # Add optional fields
            if task_info.get('deadline'):
                task['deadline'] = task_info['deadline']
            if task_info.get('priority'):
                task['priority'] = task_info['priority']
            if task_info.get('tags'):
                task['tags'] = task_info['tags']
            
            # Add to both in-memory list and vector store
            self.tasks.append(task)
            print(f"Created task: {task}")
            self._add_task_to_vector_store(task)
            created_tasks.append(task)
        
        state["created_tasks"] = created_tasks
        state["tasks"] = self.tasks
        return state
    
    def _update_tasks(self, state: TaskState2) -> TaskState2:
        """Update existing tasks"""
        print(f"Updating tasks: {state['user_input']}")
        updated_tasks = []
        # Implementation for updating tasks would go here
        user_input = state["user_input"].lower()
        
        # Search for tasks to update using vector similarity
        potential_tasks = self._search_tasks(user_input, k=10)
        
        for task in potential_tasks:
            if not task.get('completed', False):
                # Update logic based on user input
                updated = False
                original_task = task.copy()
                
                # Update deadline if mentioned
                new_deadline = self._extract_deadline(user_input)
                if new_deadline and new_deadline != task.get('deadline'):
                    task['deadline'] = new_deadline
                    updated = True
                
                # Update priority if mentioned
                if 'high priority' in user_input or 'urgent' in user_input:
                    task['priority'] = 'high'
                    updated = True
                elif 'low priority' in user_input:
                    task['priority'] = 'low'
                    updated = True
                
                if updated:
                    task['updated_at'] = datetime.now().isoformat()
                    
                    # Update in both memory and vector store
                    for i, mem_task in enumerate(self.tasks):
                        if mem_task['id'] == task['id']:
                            self.tasks[i] = task
                            break
                    
                    self._update_task_in_vector_store(task)
                    updated_tasks.append(task)
                    break  # Update only first matching task
        
        state["updated_tasks"] = updated_tasks
        state["tasks"] = self.tasks
        return state
    
    def _complete_tasks(self, state: TaskState2) -> TaskState2:
        """Mark tasks as complete"""
        print(f"Completing tasks: {state['user_input']}")
        user_input = state["user_input"].lower()
        completed_tasks = []
        
        # Search for tasks to complete using vector similarity
        potential_tasks = self._search_tasks(user_input, k=10)
        
        for task in potential_tasks:
            if not task.get('completed', False):
                if task['title'].lower() in user_input or task['id'] in user_input:
                    task['completed'] = True
                    task['completed_at'] = datetime.now().isoformat()
                    
                    # Update in both memory and vector store
                    for i, mem_task in enumerate(self.tasks):
                        if mem_task['id'] == task['id']:
                            self.tasks[i] = task
                            break
                    
                    self._update_task_in_vector_store(task)
                    completed_tasks.append(task)
        
        state["updated_tasks"] = completed_tasks
        state["tasks"] = self.tasks
        return state
    
    def _generate_summary(self, state: TaskState2) -> TaskState2:
        """Generate a task summary"""
        print(f"Generating summary: {state['user_input']}")
        tasks_summary = self.get_tasks_summary()
        
        summary_prompt = """
        Based on the following task summary, create a helpful overview for the user.
        Use a friendly and encouraging tone. Group tasks by urgency and highlight any missing deadlines.
        
        Task Summary:
        {tasks_summary}
        
        Category: {category}
        """
        
        response = self.llm.invoke([
            HumanMessage(content=summary_prompt.format(
                tasks_summary=json.dumps(tasks_summary, indent=2),
                category=self.category
            ))
        ])
        
        state["response"] = response.content
        return state
    
    def _generate_response(self, state: TaskState2) -> TaskState2:
        """Generate final response to user"""
        print(f"Generating response: {state['user_input']}")
        if state.get("response"):
            return state  # Already have a response from summary
        
        context = {
            "intent": state["intent"],
            "created_tasks": state.get("created_tasks", []),
            "updated_tasks": state.get("updated_tasks", []),
            "category": self.category
        }
        
        response_prompt = f"""
        {self.role_prompt}
        
        Context: The user's intent was '{context["intent"]}' for {context["category"]} tasks.
        
        Created tasks: {len(context["created_tasks"])}
        Updated tasks: {len(context["updated_tasks"])}
        
        Original user input: {state["user_input"]}
        
        Provide a helpful response acknowledging what was done and offering next steps if appropriate.
        """
        
        response = self.llm.invoke([HumanMessage(content=response_prompt)])
        state["response"] = response.content
        return state

    def _ensure_task_id_index(self):
        """
        Ensures that a payload index exists for the 'id' field in the Qdrant collection.
        """
        try:
            # Check if the collection exists (optional, but good practice)
            # You might have this check elsewhere, or handle CollectionNotFoundError
            collection_info = self.qdrant_client.get_collection(collection_name=self.collection_name)
            print(f"Collection '{self.collection_name}' exists. Status: {collection_info.status.value}")

            # Create the payload index for 'id'
            # Using FieldIndexType.KEYWORD is appropriate for exact string matches like UUIDs
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.id",  # the nested structure is within metadata
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            print(f"Payload index for 'id' created or already exists in collection '{self.collection_name}'.")
        except Exception as e:
            print(f"Error ensuring 'id' index for collection '{self.collection_name}': {e}")
            # Handle the error appropriately, e.g., log it, raise it, or exit

    def get_task_by_id(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a task by its ID, checking in-memory first, then Qdrant.
        """
        # Check in-memory tasks first
        for task in self.tasks:
            if task.get('id') == task_id:
                print(f"Task {task_id} found in in-memory list.")
                return task

        # If not in memory, search Qdrant
        print(f"Task {task_id} not in memory. Search collection {self.collection_name}")
        try:
            search_results = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="id",
                            match=models.MatchValue(value=task_id)
                        )
                    ]
                ),
                limit=1,
                with_payload=True, # We need the full payload (task data)
                with_vectors=False
            )
            
            if search_results and search_results[0] and search_results[0][0].payload:
                found_task = search_results[0][0].payload
                print(f"Task {task_id} found in Qdrant.")
                # Optionally add to in-memory list if not already there
                if found_task not in self.tasks:
                    self.tasks.append(found_task)
                return found_task
            else:
                print(f"Task {task_id} not found in Qdrant.")
                return None
        except Exception as e:
            print(f"Error searching for task {task_id} in Qdrant: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def process_message(self, user_input: str):
        """Process user message through the LangGraph workflow"""
        print(f"Processing message: {user_input}")
        initial_state = TaskState2(
            messages=[],
            user_input=user_input,
            tasks=self.tasks,
            created_tasks=[],
            updated_tasks=[],
            intent="",
            category=self.category,
            user_id=self.user_id,
            response="",
            extracted_info={}
        )
        
        # Run the workflow
        config = {"configurable": {"thread_id": f"{self.user_id}_{self.category}"}}
        final_state = await self.workflow.ainvoke(initial_state, config)
        
        # Update conversation history
        self.conversation_history.append(HumanMessage(content=user_input))
        self.conversation_history.append(AIMessage(content=final_state["response"]))
        
        return final_state["response"], final_state.get("created_tasks", [])
    
    # Enhanced helper methods with vector store integration
    def get_tasks_summary(self):
        # Get fresh tasks from vector store
        print(f"Getting tasks summary")
        all_tasks = self._get_all_tasks_from_vector_store()
        
        if not all_tasks:
            return "No tasks currently tracked."
        
        now = datetime.now()
        today = now.date()
        this_week_end = today + timedelta(days=(6 - today.weekday()))
        
        overdue = []
        due_today = []
        due_this_week = []
        future = []
        no_deadline = []
        
        for task in all_tasks:
            if task.get('completed', False):
                continue
                
            deadline = task.get('deadline')
            if not deadline:
                no_deadline.append(task)
                continue
            
            try:
                deadline_date = datetime.fromisoformat(deadline).date()
                if deadline_date < today:
                    overdue.append(task)
                elif deadline_date == today:
                    due_today.append(task)
                elif deadline_date <= this_week_end:
                    due_this_week.append(task)
                else:
                    future.append(task)
            except:
                no_deadline.append(task)
        
        return {
            'overdue': overdue,
            'due_today': due_today,
            'due_this_week': due_this_week,
            'future': future,
            'no_deadline': no_deadline
        }
    
    def search_tasks_by_content(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search tasks by content using vector similarity"""
        return self._search_tasks(query, k=limit)
    
    def add_task_manual(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Manually add a task (useful for direct API calls)"""
        task['id'] = task.get('id', str(uuid.uuid4()))
        task['created_at'] = task.get('created_at', datetime.now().isoformat())
        task['category'] = self.category
        task['user_id'] = self.user_id
        
        self.tasks.append(task)
        self._add_task_to_vector_store(task)
        return task
    
    def update_task_manual(self, task_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Manually update a task"""
        # Find task in memory
        task = None
        for i, t in enumerate(self.tasks):
            if t['id'] == task_id:
                task = t
                break
        
        if not task:
            # Try to find in vector store
            all_tasks = self._get_all_tasks_from_vector_store()
            for t in all_tasks:
                if t['id'] == task_id:
                    task = t
                    self.tasks.append(task)
                    break
        
        if task:
            task.update(updates)
            task['updated_at'] = datetime.now().isoformat()
            
            # Update in vector store
            self._update_task_in_vector_store(task)
            return task
        
        return None
    
    def delete_task(self, task_id: str) -> bool:
        """Delete a task from both memory and vector store"""
        print(f"Attempting to delete task ID: {task_id} in collection {self.collection_name}")
        task_found = False
        
        # First check in-memory tasks to get the document ID if available
        doc_id = self.task_id_to_doc_id.get(task_id)
        
        # Try to delete from vector store if we have a document ID
        if doc_id:
            try:
                print(f"Deleting from vector store using doc_id {doc_id}")
                # First try to delete using the document ID if we have it
                self.qdrant_client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(
                        points=[doc_id],
                    ),
                )
                print(f"Successfully deleted task {task_id} from vector store using doc_id {doc_id}")
                task_found = True
            except Exception as e:
                print(f"Error deleting task using doc_id: {str(e)}")
        else:
            try:
                print(f"Delete using task_id: {task_id}")

                """ scroll_result = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    limit=10,  # Number of points per batch
                    with_payload=True,
                )
                print(f"Scroll result: {scroll_result}")"""

                # Search for any document with this task_id in its payload
                search_results = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.id",  # the nested structure is within metadata
                                match=models.MatchValue(value=task_id)
                            )
                        ]
                    ),
                    limit=1,
                    with_payload=True
                )
                print(f"Search results: {search_results}")
                if search_results and search_results[0]:
                    point_id = search_results[0][0].id
                    self.qdrant_client.delete(
                        collection_name=self.collection_name,
                        points_selector=models.PointIdsList(
                            points=[point_id],
                        ),
                    )
                    print(f"Successfully deleted task {task_id} from vector store using task_id search")
                    task_found = True
            except Exception as e2:
                print(f"Error in delete attempt: {str(e2)}")
        
        # Remove from in-memory storage
        initial_count = len(self.tasks)
        self.tasks = [t for t in self.tasks if t.get('id') != task_id]
        task_was_in_memory = len(self.tasks) < initial_count
        print(f"Task delete {task_found} {task_was_in_memory}")
        # Remove from task_id_to_doc_id mapping
        if task_id in self.task_id_to_doc_id:
            del self.task_id_to_doc_id[task_id]
        
        # Return True if the task was found and deleted from either storage
        return task_was_in_memory or task_found
    
    def _extract_deadline(self, text: str) -> str:
        """Extract deadline from text - basic implementation"""
        text_lower = text.lower()
        now = datetime.now()
        
        if "end of day today" in text_lower or "by today" in text_lower:
            return now.replace(hour=23, minute=59, second=59).isoformat()
        elif "next monday" in text_lower:
            days_ahead = 7 - now.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return (now + timedelta(days=days_ahead)).isoformat()
        elif "next tuesday" in text_lower:
            days_ahead = 1 - now.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return (now + timedelta(days=days_ahead)).isoformat()
        elif "this weekend" in text_lower:
            days_ahead = 5 - now.weekday()  # Saturday
            if days_ahead < 0:
                days_ahead += 7
            return (now + timedelta(days=days_ahead)).isoformat()
        
        return None

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get all tasks, refreshed from vector store"""
        self.tasks = self._get_all_tasks_from_vector_store()
        return self.tasks

    def mark_task_complete(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Mark a task as complete"""
        return self.update_task_manual(task_id, {
            'completed': True,
            'completed_at': datetime.now().isoformat()
        })

    def get_qdrant_collection_info(self) -> Dict[str, Any]:
        """Get information about the Qdrant collection"""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status
            }
        except Exception as e:
            return {"error": str(e)}

class TaskManager2:
    def __init__(self):
        self.assistants = {}
        self.qdrant_url = os.getenv("QDRANT_URL")
        print(f"QDRANT_URL: {self.qdrant_url}")

        # Personal assistant configuration
        personal_role = """You are a friendly and organized personal task assistant powered by LangGraph workflows. Your main focus is helping users stay on top of their personal tasks and commitments through structured processing. Specifically:

- Help track and organize personal tasks using advanced workflow capabilities
- When providing a 'todo summary':
  1. List all current tasks grouped by deadline (overdue, today, this week, future)
  2. Highlight any tasks missing deadlines and gently encourage adding them
  3. Note any tasks that seem important but lack time estimates
- Proactively ask for deadlines when new tasks are added without them
- Maintain a supportive tone while helping the user stay accountable
- Help prioritize tasks based on deadlines and importance
- Use intelligent task extraction and intent recognition

Your communication style should be encouraging and helpful, never judgmental. 

When tasks are missing deadlines, respond with something like "I notice [task] doesn't have a deadline yet. Would you like to add one to help us track it better?"""

        # Work assistant configuration
        work_role = """You are a focused and efficient work task assistant powered by LangGraph workflows. 

Your main focus is helping users manage their work commitments with realistic timeframes through structured processing. 

Specifically:

- Help track and organize work tasks using advanced workflow capabilities
- When providing a 'todo summary':
  1. List all current tasks grouped by deadline (overdue, today, this week, future)
  2. Highlight any tasks missing deadlines and gently encourage adding them
  3. Note any tasks that seem important but lack time estimates
- When discussing new tasks, suggest that the user provide realistic time-frames based on task type:
  • Developer Relations features: typically 1 day
  • Course lesson reviews/feedback: typically 2 days
  • Documentation sprints: typically 3 days
- Help prioritize tasks based on deadlines and team dependencies
- Maintain a professional tone while helping the user stay accountable
- Use intelligent task extraction and intent recognition

Your communication style should be supportive but practical. 

When tasks are missing deadlines, respond with something like "I notice [task] doesn't have a deadline yet. Based on similar tasks, this might take [suggested timeframe]. Would you like to set a deadline with this in mind?"""

        self.assistants['personal'] = TaskAssistant2(personal_role, 'personal', 'lance', self.qdrant_url)
        self.assistants['work'] = TaskAssistant2(work_role, 'work', 'lance', self.qdrant_url)
    
    def get_assistant(self, category: str) -> TaskAssistant2:
        return self.assistants.get(category)

# Usage example
async def main():
    task_manager = TaskManager2()
    
    # Get work assistant
    work_assistant = task_manager.get_assistant('work')
    
    # Create work tasks
    print("=== Creating Work Tasks with LangGraph ===")
    response, created_tasks = await work_assistant.process_message(
        "Create or update few ToDos: 1) Re-film Module 6, lesson 5 by end of day today. 2) Update audioUX by next Monday."
    )
    print("Assistant:", response)
    print(f"Created {len(created_tasks)} tasks")
    
    response, created_tasks = await work_assistant.process_message(
        "Create another ToDo: Finalize set of report generation tutorials."
    )
    print("Assistant:", response)
    
    response, _ = await work_assistant.process_message(
        "OK, for this task let's get it done by next Tuesday."
    )
    print("Assistant:", response)
    
    # Get personal assistant
    personal_assistant = task_manager.get_assistant('personal')
    
    # Create personal tasks
    print("\n=== Creating Personal Tasks with LangGraph ===")
    response, created_tasks = await personal_assistant.process_message(
        "Create ToDos: 1) Check on swim lessons for the baby this weekend. 2) For winter travel, check AmEx points."
    )
    print("Assistant:", response)
    
    # Get todo summary
    print("\n=== Personal Todo Summary ===")
    response, _ = await personal_assistant.process_message("Give me a todo summary.")
    print("Assistant:", response)
    
    # Test task completion
    print("\n=== Completing a Task ===")
    if personal_assistant.tasks:
        task_to_complete = personal_assistant.tasks[0]
        response, _ = await personal_assistant.process_message(
            f"Mark task '{task_to_complete['title']}' as complete"
        )
        print("Assistant:", response)
    
    # Test semantic search capabilities
    print("\n=== Semantic Search Test ===")
    search_results = work_assistant.search_tasks_by_content("video filming module")
    print(f"Found {len(search_results)} tasks matching 'video filming module':")
    for task in search_results[:3]:
        print(f"- {task['title']} (Score-based match)")
    
    # Test task search and update
    print("\n=== Task Search and Update ===")
    response, _ = work_assistant.process_message(
        "Find the task about filming and update its priority to high"
    )
    print("Assistant:", response)
    
    # Show vector store persistence
    print("\n=== Vector Store Persistence Info ===")
    print(f"Work tasks in vector store: {len(work_assistant._get_all_tasks_from_vector_store())}")
    print(f"Personal tasks in vector store: {len(personal_assistant._get_all_tasks_from_vector_store())}")
    print(f"Task-to-Document mappings: {len(work_assistant.task_id_to_doc_id)}")
    
    # Print workflow visualization info
    print("\n=== LangGraph Workflow Info ===")
    print("Work Assistant Workflow Nodes:", list(work_assistant.workflow.graph.nodes.keys()))
    print("Personal Assistant Workflow Nodes:", list(personal_assistant.workflow.graph.nodes.keys()))

# Run the example
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
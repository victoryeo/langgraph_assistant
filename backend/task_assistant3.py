from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
#from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector # Import PGVector
from langchain_community.vectorstores.pgvector import DistanceStrategy
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime, timedelta
from typing import Dict, List, Any, TypedDict, Annotated, Optional
from dotenv import load_dotenv
import operator
import psycopg2 # For direct DB operations if needed, or rely on PGVector
import os
import re
import json
import uuid

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["SUPABASE_URL"] = os.getenv("SUPABASE_URL")
os.environ["SUPABASE_KEY"] = os.getenv("SUPABASE_KEY")
os.environ["SUPABASE_DB_CONNECTION_STRING"] = os.getenv("SUPABASE_DB_CONNECTION_STRING")

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
class TaskAssistant3:
    def __init__(self, role_prompt: str, category: str, user_id: str):
        self.role_prompt = role_prompt
        self.category = category
        self.user_id = user_id
        self.db_connection_string = os.getenv("SUPABASE_DB_CONNECTION_STRING")
        if not self.db_connection_string:
            raise ValueError("SUPABASE_DB_CONNECTION_STRING environment variable is not set.")

        # LLM setup
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            self.llm = ChatGroq(temperature=0.7, model_name="llama-3.3-70b-versatile")
        else:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                #self.llm = ChatOpenAI(model="gpt-4")
                print("OpenAI API key is set")
            else:
                raise ValueError("Neither GROQ_API_KEY nor OPENAI_API_KEY is set")

        # Task tracking
        self.tasks = []  # In-memory storage, kept for immediate operations
        self.conversation_history = []
        
        # Setup embeddings for vector store
        self.embeddings = self._setup_embeddings()
        
        # Initialize PGVector client and table
        self.collection_name = f"tasks_{category}_{user_id}" # Used as table name
        
        # supabase initialization
        #url: str = os.environ.get("SUPABASE_URL")
        #key: str = os.environ.get("SUPABASE_KEY")
        #supabase: Client = create_client(url, key)

        # Initialize PGVector vector store
        self.vector_store = PGVector(
            collection_name=self.collection_name,
            connection_string=self.db_connection_string,
            embedding_function=self.embeddings,
            distance_strategy=DistanceStrategy.COSINE, # Align with Qdrant's COSINE
            use_jsonb=True # <--- ADD THIS LINE TO EXPLICITLY SET JSONB
        )
        print(f"DEBUG: PGVector initialized with collection_name: {self.collection_name}")
        
        # Create LangGraph workflow
        self.workflow = self._create_workflow()
        
    def _setup_embeddings(self):
        """Setup embeddings for vector store"""
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    def _task_to_document(self, task: Dict[str, Any]) -> Document:
        """Convert a task dictionary to a Document for vector storage"""
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
        metadata['searchable_content'] = content # Redundant, but kept for consistency
        
        # PGVector typically handles the UUID internally if not provided,
        # but we use task['id'] as the primary key for direct control.
        return Document(
            page_content=content,
            metadata=metadata
        )
    
    def _add_task_to_vector_store(self, task: Dict[str, Any]) -> str:
        """Add a task to the vector store and return the document ID (primary key)"""
        document = self._task_to_document(task)
        
        try:
            # PGVector's add_documents can directly handle this.
            # It will use the 'id' from metadata if specified, or generate one.
            doc_ids = self.vector_store.add_documents([document])
            doc_id = doc_ids[0] if doc_ids else None # Get the first ID
            if doc_id:
                print(f"DEBUG: Successfully added task to vector store with ID: {doc_id}")
            else:
                print(f"DEBUG: PGVector.add_documents returned no ID for task: {task.get('id')}. This is unexpected if no error occurred.")
            return doc_id
        except Exception as e:
            print(f"ERROR: Failed to add task to vector store: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for more details
            return None # Indicate failure

    def _update_task_in_vector_store(self, task: Dict[str, Any]):
        task_id = str(task['id'])
        print(f"DEBUG: _update_task_in_vector_store START for task_id: {task_id}")
        
        # --- Direct SQL Delete Approach (if LangChain's delete fails) ---
        try:
            conn = psycopg2.connect(self.db_connection_string)
            cur = conn.cursor()
            
            # Get the collection UUID (needed to filter rows belonging to this logical collection)
            cur.execute(f"SELECT uuid FROM langchain_pg_collection WHERE name = %s;", (self.collection_name,))
            collection_uuid_row = cur.fetchone()
            
            if collection_uuid_row:
                collection_uuid = collection_uuid_row[0]
                print(f"DEBUG: Attempting direct SQL DELETE for collection {self.collection_name} ({collection_uuid}) and task_id {task_id}.")
                
                delete_sql = f"DELETE FROM langchain_pg_embedding WHERE collection_id = %s AND cmetadata->>'id' = %s;"
                cur.execute(delete_sql, (str(collection_uuid), task_id))
                rows_deleted = cur.rowcount
                conn.commit() # <<< Explicit COMMIT is crucial for direct SQL
                print(f"DEBUG: Direct SQL DELETE completed. Rows deleted: {rows_deleted} for task ID {task_id}.")
                if rows_deleted == 0:
                    print(f"WARNING: Direct SQL delete found 0 rows for task ID {task_id}. Check ID, collection, or if already deleted.")
            else:
                print(f"WARNING: Collection '{self.collection_name}' not found during direct SQL delete attempt. No delete performed.")

            cur.close()
            conn.close()

        except Exception as e:
            print(f"ERROR: Exception during direct SQL delete for task {task_id}: {e}")
            import traceback
            traceback.print_exc()
            # If deletion failed, you might still want to try to add the new version.
        
        # --- Add the updated document ---
        print(f"DEBUG: Calling _add_task_to_vector_store for task_id: {task_id} after delete attempt.")
        self._add_task_to_vector_store(task)
        print(f"DEBUG: _update_task_in_vector_store END for task_id: {task_id}")

    def _search_tasks(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search tasks using vector similarity in PGVector"""
        try:
            results = self.vector_store.similarity_search(query, k=k)
            tasks = []
            for doc in results:
                # PGVector's results typically return Document objects with metadata
                if 'id' in doc.metadata: # Ensure your original task ID is in metadata
                    tasks.append(doc.metadata)
                else:
                    print(f"WARNING: Document found without 'id' in metadata: {doc.metadata}")
            print(f"DEBUG: Found {len(tasks)} tasks via similarity search.")
            return tasks
        except Exception as e:
            print(f"ERROR: Search error in LangChain PGVector: {e}")
            import traceback
            traceback.print_exc()
            return [t for t in self.tasks if t.get('user_id') == self.user_id and t.get('category') == self.category] # Fallback to in-memory, filtered

    def _get_all_tasks_from_vector_store(self) -> List[Dict[str, Any]]:
        """Get all tasks from PGVector, filtered by category and user_id"""
        try:
            conn = psycopg2.connect(self.db_connection_string)
            cur = conn.cursor()

            # First, find the collection_id for your logical collection_name
            cur.execute(f"SELECT uuid FROM langchain_pg_collection WHERE name = %s;", (self.collection_name,))
            collection_uuid_row = cur.fetchone()
            
            if not collection_uuid_row:
                print(f"WARNING: Collection '{self.collection_name}' not found in langchain_pg_collection. No tasks to retrieve.")
                cur.close()
                conn.close()
                return []

            collection_uuid = collection_uuid_row[0]
            
            # Then, retrieve documents from langchain_pg_embedding for that collection_id
            # The metadata is stored in the 'cmetadata' column
            cur.execute(f"SELECT cmetadata FROM langchain_pg_embedding WHERE collection_id = %s;", (str(collection_uuid),))
            
            all_tasks_metadata = cur.fetchall()
            cur.close()
            conn.close()
            
            tasks = []
            for row in all_tasks_metadata:
                metadata = row[0] 
                # The 'cmetadata' column in langchain_pg_embedding IS your metadata
                # So it should contain 'id', 'category', 'user_id' directly.
                if (metadata.get('category') == self.category and 
                    metadata.get('user_id') == self.user_id):
                    tasks.append(metadata)
                else:
                    print(f"DEBUG: Skipping task from DB due to category/user_id mismatch in cmetadata: {metadata}")
            
            print(f'DEBUG: Found {len(tasks)} total tasks in LangChain PGVector for {self.category}/{self.user_id}')
            return tasks
        except Exception as e:
            print(f"ERROR: Error getting all tasks from LangChain PGVector (direct query): {e}")
            import traceback
            traceback.print_exc()
            return [t for t in self.tasks if t.get('user_id') == self.user_id and t.get('category') == self.category]
        
    def _create_workflow(self):
        """Create the LangGraph workflow for task processing"""
        workflow = StateGraph(TaskState2)
        
        workflow.add_node("analyze_intent", self._analyze_intent)
        workflow.add_node("extract_task_info", self._extract_task_info)
        workflow.add_node("create_tasks", self._create_tasks)
        workflow.add_node("update_tasks", self._update_tasks)
        workflow.add_node("complete_tasks", self._complete_tasks)
        workflow.add_node("generate_summary", self._generate_summary)
        workflow.add_node("generate_response", self._generate_response)
        
        workflow.set_entry_point("analyze_intent")
        
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
        
        workflow.add_conditional_edges(
            "extract_task_info",
            self._route_by_action,
            {
                "create": "create_tasks",
                "update": "update_tasks"
            }
        )
        
        workflow.add_edge("create_tasks", "generate_response")
        workflow.add_edge("update_tasks", "generate_response")
        workflow.add_edge("complete_tasks", "generate_response")
        workflow.add_edge("generate_summary", "generate_response")
        
        workflow.add_edge("generate_response", END)
        
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def _analyze_intent(self, state: TaskState2) -> TaskState2:
        """Analyze user intent from the input"""
        user_input = state["user_input"].lower()
        
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
        user_input = state["user_input"].lower()
        
        # Search for tasks to update using vector similarity
        potential_tasks = self._search_tasks(user_input, k=10)
        
        for task in potential_tasks:
            if not task.get('completed', False):
                updated = False
                # original_task = task.copy() # not used
                
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
                # A more robust check might be needed here to confirm the exact task,
                # perhaps with an LLM call or ID extraction.
                # For simplicity, if title or ID is in input, we consider it a match.
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
        
        state["updated_tasks"] = completed_tasks # Reusing updated_tasks for completed tasks
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

    def get_task_by_id(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a task by its ID, checking in-memory first, then LangChain PGVector.
        """
        # Check in-memory tasks first
        for task in self.tasks:
            if task.get('id') == task_id:
                print(f"DEBUG: Task {task_id} found in in-memory list.")
                return task

        # If not in memory, search LangChain PGVector directly using the ID in metadata
        print(f"DEBUG: Task {task_id} not in memory. Searching LangChain PGVector for collection: {self.collection_name}")
        try:
            conn = psycopg2.connect(self.db_connection_string)
            cur = conn.cursor()
            
            # Find the collection_id for your logical collection_name
            cur.execute(f"SELECT uuid FROM langchain_pg_collection WHERE name = %s;", (self.collection_name,))
            collection_uuid_row = cur.fetchone()
            
            if not collection_uuid_row:
                print(f"WARNING: Collection '{self.collection_name}' not found during get_task_by_id.")
                cur.close()
                conn.close()
                return None
            
            collection_uuid = collection_uuid_row[0]

            # Query langchain_pg_embedding table, filtering by collection_id and metadata->>'id'
            cur.execute(f"SELECT cmetadata FROM langchain_pg_embedding WHERE collection_id = %s AND cmetadata->>'id' = %s LIMIT 1;", 
                        (str(collection_uuid), task_id))
            
            result = cur.fetchone()
            cur.close()
            conn.close()

            if result and result[0]:
                found_task = result[0] # cmetadata is the first (and only) column selected
                print(f"DEBUG: Task {task_id} found in LangChain PGVector.")
                # Optionally add to in-memory list if not already there
                if found_task not in self.tasks:
                    self.tasks.append(found_task)
                return found_task
            else:
                print(f"DEBUG: Task {task_id} not found in LangChain PGVector.")
                return None
        except Exception as e:
            print(f"ERROR: Error searching for task {task_id} in LangChain PGVector: {e}")
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
        
        config = {"configurable": {"thread_id": f"{self.user_id}_{self.category}"}}
        final_state = await self.workflow.ainvoke(initial_state, config)
        
        self.conversation_history.append(HumanMessage(content=user_input))
        self.conversation_history.append(AIMessage(content=final_state["response"]))
        
        return final_state["response"], final_state.get("created_tasks", [])
    
    def get_tasks_summary(self):
        print(f"Getting tasks summary from PGVector")
        all_tasks = self._get_all_tasks_from_vector_store() # Fetches tasks from DB
        
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
        """Search tasks by content using vector similarity in PGVector"""
        return self._search_tasks(query, k=limit)
    
    def add_task_manual(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Manually add a task (useful for direct API calls)"""
        task['id'] = task.get('id', str(uuid.uuid4()))
        task['created_at'] = task.get('created_at', datetime.now().isoformat())
        task['category'] = self.category
        task['user_id'] = self.user_id
        
        self.tasks.append(task) # Add to in-memory
        self._add_task_to_vector_store(task) # Add to DB
        return task
    
    def update_task_manual(self, task_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Manually update a task"""
        # Fetch the task, prioritize from memory, then DB
        task = None
        for i, t in enumerate(self.tasks):
            if t['id'] == task_id:
                task = t
                break
        
        if not task:
            task = self.get_task_by_id(task_id) # Use the unified get_task_by_id
        
        if task:
            task.update(updates)
            task['updated_at'] = datetime.now().isoformat()
            
            # Update in memory (if found there)
            for i, t in enumerate(self.tasks):
                if t['id'] == task_id:
                    self.tasks[i] = task
                    break
            else: # If not found in memory, add it
                self.tasks.append(task)
            
            # Update in vector store (DB)
            self._update_task_in_vector_store(task)
            return task
        
        return None
    
    def delete_task(self, task_id: str) -> bool:
        """
        Delete a task from both memory and LangChain PGVector.
        This method will also use the internal ID lookup for reliability.
        """
        print(f"DEBUG: delete_task START for task_id: {task_id}")
        task_deleted_from_db = False
        
        try:
            conn = psycopg2.connect(self.db_connection_string)
            cur = conn.cursor()
            
            cur.execute(f"SELECT uuid FROM langchain_pg_collection WHERE name = %s;", (self.collection_name,))
            collection_uuid_row = cur.fetchone()
            
            if collection_uuid_row:
                collection_uuid = collection_uuid_row[0]
                print(f"DEBUG: Attempting direct SQL DELETE for collection {self.collection_name} ({collection_uuid}) and task_id {task_id}.")
                
                delete_sql = f"DELETE FROM langchain_pg_embedding WHERE collection_id = %s AND cmetadata->>'id' = %s;"
                cur.execute(delete_sql, (str(collection_uuid), task_id))
                rows_deleted = cur.rowcount
                conn.commit() # <<< Explicit COMMIT is crucial for direct SQL
                print(f"DEBUG: Direct SQL DELETE completed. Rows deleted: {rows_deleted} for task ID {task_id}.")
                if rows_deleted == 0:
                    print(f"WARNING: Direct SQL delete found 0 rows for task ID {task_id}. Check ID, collection, or if already deleted.")
            else:
                print(f"WARNING: Collection '{self.collection_name}' not found during direct SQL delete attempt. No delete performed.")

            cur.close()
            conn.close()
            task_deleted_from_db = True
                
        except Exception as e:
            print(f"ERROR: Exception during DB deletion for task {task_id}: {e}")
            import traceback
            traceback.print_exc()
        
        # Remove from in-memory storage
        initial_count = len(self.tasks)
        self.tasks = [t for t in self.tasks if t.get('id') != task_id]
        task_was_in_memory = len(self.tasks) < initial_count
        
        print(f"DEBUG: delete_task END. DB deleted={task_deleted_from_db}, In-memory deleted={task_was_in_memory}")
        
        return task_deleted_from_db or task_was_in_memory

    def _extract_deadline(self, text: str) -> str:
        """Extract deadline from text - basic implementation"""
        text_lower = text.lower()
        now = datetime.now()
        
        # Example basic extraction (can be expanded)
        if "today" in text_lower:
            return now.date().isoformat()
        elif "tomorrow" in text_lower:
            return (now + timedelta(days=1)).date().isoformat()
        elif "next week" in text_lower:
            # Assuming end of next week
            next_week_end = now + timedelta(days=(6 - now.weekday() + 7))
            return next_week_end.date().isoformat()
        
        # Simple regex for YYYY-MM-DD or MM/DD/YYYY (can be improved)
        date_match = re.search(r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}', text)
        if date_match:
            try:
                # Attempt to parse
                date_str = date_match.group(0)
                if '-' in date_str:
                    return datetime.strptime(date_str, '%Y-%m-%d').date().isoformat()
                elif '/' in date_str:
                    return datetime.strptime(date_str, '%m/%d/%Y').date().isoformat()
            except ValueError:
                pass
        
        return "" # No deadline found

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

class TaskManager3:
    def __init__(self):
        # A nested dictionary to store assistants:
        # self.assistants = {
        #   'user_lance_email@example.com': {
        #     'work': TaskAssistant3_instance_for_work,
        #     'personal': TaskAssistant3_instance_for_personal
        #   },
        # }
        self.assistants: Dict[str, Dict[str, TaskAssistant3]] = {}
        self.qdrant_url = os.getenv("QDRANT_URL")
        print(f"QDRANT_URL: {self.qdrant_url}")

        # Personal assistant configuration
        self.personal_role = """You are a friendly and organized personal task assistant powered by LangGraph workflows. Your main focus is helping users stay on top of their personal tasks and commitments through structured processing. Specifically:

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
        self.work_role = """You are a focused and efficient work task assistant powered by LangGraph workflows. 

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

    def get_assistant(self, category: str, user_id: str) -> TaskAssistant3:
        """
        Retrieves a TaskAssistant instance for a given user and category.
        If it doesn't exist, a new one is created.
        """
        # Ensure the user's dictionary exists
        if user_id not in self.assistants:
            self.assistants[user_id] = {}
            print(f"INFO: New user '{user_id}' detected. Initializing assistant dictionary.")
        
        # Check if the specific assistant exists for this user/category
        if category not in self.assistants[user_id]:
            print(f"INFO: Creating new '{category}' assistant for user '{user_id}'.")
            
            # Personal assistant configuration
            if category == 'personal':
                role_prompt = self.personal_role
            # Work assistant configuration
            elif category == 'work':
                role_prompt = self.work_role
            else:
                raise ValueError(f"Unknown assistant category: {category}")

            # Instantiate and store the new assistant, passing the user_id
            self.assistants[user_id][category] = TaskAssistant3(
                role_prompt=role_prompt,
                category=category,
                user_id=user_id # Crucially, pass the unique user ID
            )
            
        return self.assistants[user_id][category]

# Usage example
async def main():
    task_manager = TaskManager3()
    
    # Get work assistant
    current_user_id = 'lance_doe_123'
    print(f"=== Getting Work Assistant for User: {current_user_id} ===")
    work_assistant = task_manager.get_assistant('work', user_id=current_user_id)
    
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
    # --- Get and use the personal assistant for the same user ---
    print(f"\n=== Getting Personal Assistant for User: {current_user_id} ===")
    personal_assistant = task_manager.get_assistant('personal', user_id=current_user_id)
    
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
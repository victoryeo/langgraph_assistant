from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime, timedelta
from typing import Dict, List, Any, TypedDict, Annotated
import json
import uuid
import os
from dotenv import load_dotenv
import operator
import re

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

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
    def __init__(self, role_prompt: str, category: str, user_id: str):
        self.role_prompt = role_prompt
        self.category = category
        self.user_id = user_id
        
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
        
        self.tasks = []  # In-memory storage
        self.conversation_history = []
        
        # Create LangGraph workflow
        self.workflow = self._create_workflow()
        
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
        extraction_prompt = """
        Extract task information from the user input. Return a JSON object with the following structure:
        {
            "tasks": [
                {
                    "title": "task title",
                    "description": "optional description",
                    "deadline": "ISO format datetime if mentioned",
                    "priority": "high/medium/low if mentioned",
                    "tags": ["tag1", "tag2"] if any tags mentioned
                }
            ]
        }
        
        User input: {user_input}
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=extraction_prompt.format(user_input=state["user_input"]))])
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                extracted_info = json.loads(json_match.group())
            else:
                # Fallback to basic extraction
                extracted_info = self._basic_task_extraction(state["user_input"])
            
            state["extracted_info"] = extracted_info
        except Exception as e:
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
            
            self.tasks.append(task)
            created_tasks.append(task)
        
        state["created_tasks"] = created_tasks
        state["tasks"] = self.tasks
        return state
    
    def _update_tasks(self, state: TaskState2) -> TaskState2:
        """Update existing tasks"""
        updated_tasks = []
        # Implementation for updating tasks would go here
        # For now, just return the state
        state["updated_tasks"] = updated_tasks
        return state
    
    def _complete_tasks(self, state: TaskState2) -> TaskState2:
        """Mark tasks as complete"""
        user_input = state["user_input"].lower()
        completed_tasks = []
        
        # Simple completion logic - look for task IDs or titles
        for task in self.tasks:
            if not task.get('completed', False):
                if task['title'].lower() in user_input or task['id'] in user_input:
                    task['completed'] = True
                    task['completed_at'] = datetime.now().isoformat()
                    completed_tasks.append(task)
        
        state["updated_tasks"] = completed_tasks
        state["tasks"] = self.tasks
        return state
    
    def _generate_summary(self, state: TaskState2) -> TaskState2:
        """Generate a task summary"""
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
    
    async def process_message(self, user_input: str):
        """Process user message through the LangGraph workflow"""
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
    
    # Keep existing helper methods
    def get_tasks_summary(self):
        if not self.tasks:
            return "No tasks currently tracked."
        
        now = datetime.now()
        today = now.date()
        this_week_end = today + timedelta(days=(6 - today.weekday()))
        
        overdue = []
        due_today = []
        due_this_week = []
        future = []
        no_deadline = []
        
        for task in self.tasks:
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

    def get_all_tasks(self):
        return self.tasks

    def mark_task_complete(self, task_id: str):
        for task in self.tasks:
            if task['id'] == task_id:
                task['completed'] = True
                task['completed_at'] = datetime.now().isoformat()
                return task
        return None

class TaskManager2:
    def __init__(self):
        self.assistants = {}
        
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

        self.assistants['personal'] = TaskAssistant2(personal_role, 'personal', 'lance')
        self.assistants['work'] = TaskAssistant2(work_role, 'work', 'lance')
    
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
    
    # Print workflow visualization info
    print("\n=== LangGraph Workflow Info ===")
    print("Work Assistant Workflow Nodes:", list(work_assistant.workflow.graph.nodes.keys()))
    print("Personal Assistant Workflow Nodes:", list(personal_assistant.workflow.graph.nodes.keys()))

# Run the example
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
import uuid
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# NOTE: langchain task assistant
class TaskAssistant:
    def __init__(self, role_prompt: str, category: str, user_id: str):
        self.role_prompt = role_prompt
        self.category = category
        self.user_id = user_id
        
        # Check if GROQ_API_KEY is available, fallback to OpenAI if not
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            self.llm = ChatGroq(temperature=0.7, model_name="llama-3.3-70b-versatile")
        else:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                self.llm = ChatOpenAI(model="gpt-4")
            else:
                raise ValueError("Neither GROQ_API_KEY nor OPENAI_API_KEY is set")
                
        self.tasks = []  # In-memory storage (replace with database in production)
        self.conversation_history = []
        
    def create_prompt_template(self):
        return ChatPromptTemplate.from_messages([
            ("system", self.role_prompt),
            ("human", "Current tasks: {tasks}\n\nUser input: {user_input}")
        ])
    
    def add_task(self, task: Dict[str, Any]):
        task['id'] = str(uuid.uuid4())
        task['created_at'] = datetime.now().isoformat()
        task['category'] = self.category
        task['user_id'] = self.user_id
        self.tasks.append(task)
        return task
    
    def update_task(self, task_id: str, updates: Dict[str, Any]):
        for task in self.tasks:
            if task['id'] == task_id:
                task.update(updates)
                return task
        return None
    
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
    
    async def process_message(self, user_input: str):
        # Parse tasks from user input (basic implementation)
        tasks_summary = self.get_tasks_summary()
        created_tasks = []
        
        # Extract and create tasks before processing with LLM
        if "create" in user_input.lower() and "todo" in user_input.lower():
            created_tasks = self._extract_and_create_tasks(user_input)
        
        prompt = self.create_prompt_template()
        response = await self.llm.ainvoke(
            prompt.format_messages(
                tasks=json.dumps(tasks_summary, indent=2),
                user_input=user_input
            )
        )
        
        self.conversation_history.append(HumanMessage(content=user_input))
        self.conversation_history.append(response)
        
        return response.content, created_tasks
    
    def _extract_and_create_tasks(self, user_input: str):
        # Basic task extraction - you might want to use an LLM for better parsing
        created_tasks = []
        lines = user_input.split('\n')
        
        for line in lines:
            if any(marker in line for marker in ['1)', '2)', '3)', '4)', '5)', '-', '*']):
                task_text = line.strip()
                # Remove numbering/bullets
                for marker in ['1)', '2)', '3)', '4)', '5)', '-', '*']:
                    task_text = task_text.replace(marker, '').strip()
                
                if task_text:
                    task = {'title': task_text, 'status': 'pending'}
                    
                    # Basic deadline extraction
                    deadline = self._extract_deadline(task_text)
                    if deadline:
                        task['deadline'] = deadline
                    
                    created_task = self.add_task(task)
                    created_tasks.append(created_task)
        
        return created_tasks
    
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

class TaskManager:
    def __init__(self):
        self.assistants = {}
        
        # Personal assistant configuration
        personal_role = """You are a friendly and organized personal task assistant. Your main focus is helping users stay on top of their personal tasks and commitments. Specifically:

- Help track and organize personal tasks
- When providing a 'todo summary':
  1. List all current tasks grouped by deadline (overdue, today, this week, future)
  2. Highlight any tasks missing deadlines and gently encourage adding them
  3. Note any tasks that seem important but lack time estimates
- Proactively ask for deadlines when new tasks are added without them
- Maintain a supportive tone while helping the user stay accountable
- Help prioritize tasks based on deadlines and importance

Your communication style should be encouraging and helpful, never judgmental. 

When tasks are missing deadlines, respond with something like "I notice [task] doesn't have a deadline yet. Would you like to add one to help us track it better?"""

        # Work assistant configuration
        work_role = """You are a focused and efficient work task assistant. 

Your main focus is helping users manage their work commitments with realistic timeframes. 

Specifically:

- Help track and organize work tasks
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

Your communication style should be supportive but practical. 

When tasks are missing deadlines, respond with something like "I notice [task] doesn't have a deadline yet. Based on similar tasks, this might take [suggested timeframe]. Would you like to set a deadline with this in mind?"""

        self.assistants['personal'] = TaskAssistant(personal_role, 'personal', 'lance')
        self.assistants['work'] = TaskAssistant(work_role, 'work', 'lance')
    
    def get_assistant(self, category: str) -> TaskAssistant:
        return self.assistants.get(category)


# Usage example
async def main():
    task_manager = TaskManager()
    
    # Get work assistant
    work_assistant = task_manager.get_assistant('work')
    
    # Create work tasks
    print("=== Creating Work Tasks ===")
    response = await work_assistant.process_message(
        "Create or update few ToDos: 1) Re-film Module 6, lesson 5 by end of day today. 2) Update audioUX by next Monday."
    )
    print("Assistant:", response)
    
    response = await work_assistant.process_message(
        "Create another ToDo: Finalize set of report generation tutorials."
    )
    print("Assistant:", response)
    
    response = await work_assistant.process_message(
        "OK, for this task let's get it done by next Tuesday."
    )
    print("Assistant:", response)
    
    # Get personal assistant
    personal_assistant = task_manager.get_assistant('personal')
    
    # Create personal tasks
    print("\n=== Creating Personal Tasks ===")
    response = await personal_assistant.process_message(
        "Create ToDos: 1) Check on swim lessons for the baby this weekend. 2) For winter travel, check AmEx points."
    )
    print("Assistant:", response)
    
    # Get todo summary
    print("\n=== Personal Todo Summary ===")
    response = await personal_assistant.process_message("Give me a todo summary.")
    print("Assistant:", response)
    
    # Print all tasks for debugging
    print("\n=== All Work Tasks ===")
    for task in work_assistant.tasks:
        print(f"- {task['title']} (Deadline: {task.get('deadline', 'None')})")
    
    print("\n=== All Personal Tasks ===")
    for task in personal_assistant.tasks:
        print(f"- {task['title']} (Deadline: {task.get('deadline', 'None')})")

# Run the example
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn
from task_assistant import TaskManager
from datetime import datetime, timedelta

# Pydantic models for API requests/responses
class TaskRequest(BaseModel):
    message: str
    user_id: Optional[str] = "default_user"

class TaskResponse(BaseModel):
    success: bool
    message: str
    assistant_response: str
    created_tasks: List[Dict[str, Any]] = []

class TaskSummaryResponse(BaseModel):
    success: bool
    summary: Dict[str, Any]
    assistant_response: str

class TaskListResponse(BaseModel):
    success: bool
    tasks: List[Dict[str, Any]]

# Initialize FastAPI app
app = FastAPI(title="Task Management API", description="API for managing work and personal tasks", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize task manager
task_manager = TaskManager()

@app.get("/")
async def root():
    return {"message": "Task Management API", "endpoints": ["/work/tasks", "/personal/tasks", "/work/summary", "/personal/summary"]}

# Work Assistant Endpoints
@app.post("/work/tasks", response_model=TaskResponse)
async def create_work_tasks(request: TaskRequest):
    try:
        work_assistant = task_manager.get_assistant('work')
        if not work_assistant:
            raise HTTPException(status_code=500, detail="Work assistant not available")
        
        assistant_response, created_tasks = await work_assistant.process_message(request.message)
        
        return TaskResponse(
            success=True,
            message=f"Processed work task request. Created {len(created_tasks)} tasks.",
            assistant_response=assistant_response,
            created_tasks=created_tasks
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing work tasks: {str(e)}")

@app.get("/work/tasks", response_model=TaskListResponse)
async def get_work_tasks():
    try:
        work_assistant = task_manager.get_assistant('work')
        if not work_assistant:
            raise HTTPException(status_code=500, detail="Work assistant not available")
        
        tasks = work_assistant.get_all_tasks()
        return TaskListResponse(success=True, tasks=tasks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving work tasks: {str(e)}")

@app.post("/work/summary", response_model=TaskSummaryResponse)
async def get_work_summary(request: TaskRequest):
    try:
        work_assistant = task_manager.get_assistant('work')
        if not work_assistant:
            raise HTTPException(status_code=500, detail="Work assistant not available")
        
        summary = work_assistant.get_tasks_summary()
        assistant_response, _ = await work_assistant.process_message("Give me a todo summary.")
        
        return TaskSummaryResponse(
            success=True,
            summary=summary,
            assistant_response=assistant_response
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting work summary: {str(e)}")

# Personal Assistant Endpoints
@app.post("/personal/tasks", response_model=TaskResponse)
async def create_personal_tasks(request: TaskRequest):
    try:
        personal_assistant = task_manager.get_assistant('personal')
        if not personal_assistant:
            raise HTTPException(status_code=500, detail="Personal assistant not available")
        
        assistant_response, created_tasks = await personal_assistant.process_message(request.message)
        
        return TaskResponse(
            success=True,
            message=f"Processed personal task request. Created {len(created_tasks)} tasks.",
            assistant_response=assistant_response,
            created_tasks=created_tasks
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing personal tasks: {str(e)}")

@app.get("/personal/tasks", response_model=TaskListResponse)
async def get_personal_tasks():
    try:
        personal_assistant = task_manager.get_assistant('personal')
        if not personal_assistant:
            raise HTTPException(status_code=500, detail="Personal assistant not available")
        
        tasks = personal_assistant.get_all_tasks()
        return TaskListResponse(success=True, tasks=tasks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving personal tasks: {str(e)}")

@app.post("/personal/summary", response_model=TaskSummaryResponse)
async def get_personal_summary(request: TaskRequest):
    try:
        personal_assistant = task_manager.get_assistant('personal')
        if not personal_assistant:
            raise HTTPException(status_code=500, detail="Personal assistant not available")
        
        summary = personal_assistant.get_tasks_summary()
        assistant_response, _ = await personal_assistant.process_message("Give me a todo summary.")
        
        return TaskSummaryResponse(
            success=True,
            summary=summary,
            assistant_response=assistant_response
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting personal summary: {str(e)}")

# Task completion endpoints
@app.put("/work/tasks/{task_id}/complete")
async def complete_work_task(task_id: str):
    try:
        work_assistant = task_manager.get_assistant('work')
        if not work_assistant:
            raise HTTPException(status_code=500, detail="Work assistant not available")
        
        task = work_assistant.mark_task_complete(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return {"success": True, "message": "Task marked as complete", "task": task}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error completing task: {str(e)}")

@app.put("/personal/tasks/{task_id}/complete")
async def complete_personal_task(task_id: str):
    try:
        personal_assistant = task_manager.get_assistant('personal')
        if not personal_assistant:
            raise HTTPException(status_code=500, detail="Personal assistant not available")
        
        task = personal_assistant.mark_task_complete(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return {"success": True, "message": "Task marked as complete", "task": task}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error completing task: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "work_assistant_tasks": len(task_manager.get_assistant('work').get_all_tasks()),
        "personal_assistant_tasks": len(task_manager.get_assistant('personal').get_all_tasks())
    }

if __name__ == "__main__":
    # For development - use uvicorn in production
    uvicorn.run(
        "task_api:app",  # Replace with your filename if different
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

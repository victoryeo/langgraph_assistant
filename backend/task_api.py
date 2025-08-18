from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel, Field, validator, field_validator, EmailStr
from typing import Dict, List, Any, Optional, Union
import uvicorn
from datetime import datetime, timedelta
import re
import os
import json
from jose import JWTError, jwt
from passlib.context import CryptContext
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from authlib.integrations.starlette_client import OAuth
from starlette.middleware.sessions import SessionMiddleware
import psycopg2 # For direct DB operations if needed
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from motor.motor_asyncio import AsyncIOMotorClient  # For async support
from fastapi import Form

# Import task manager
from task_assistant3 import TaskManager3

# OAuth2 and JWT configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-here")  # Change this to a secure secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Google OAuth2 config
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")

if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
    raise ValueError("GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET must be set in environment variables")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Pydantic models for API requests/responses
class TaskRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The task message or request",
        example="Create a task to review the quarterly report"
    )
    user_id: Optional[str] = Field(
        default="default_user",
        max_length=50,
        description="User identifier",
        example="user_123"
    )
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v):
        if not v or v.isspace():
            raise ValueError('Message cannot be empty or contain only whitespace')
        
        # Remove excessive whitespace
        v = ' '.join(v.split())
        
        # Check for minimum meaningful content
        if len(v.strip()) < 2:
            raise ValueError('Message must contain at least 2 characters')

        # Check for profanity in various forms (case insensitive and with common substitutions)
        profanity_patterns = [
            r'f[\*u\$@]c[\*kq]',  # Matches f**k, f*ck, f*ck, fu*k, f**k, f*ck, etc.
            r'p[o0]rn',             # Matches porn, p0rn
            r'p[o0]rn[o0]',         # Matches porno, p0rn0
            r'wh[o0]re',            # Matches whore, wh0re
            r'sl[u\*]t',            # Matches slut, sl*t
            r'p[o0]rn[o0]gr[a@]phy', # Matches pornography, p0rn0graphy
        ]
        
        # Check if any profanity pattern matches
        for pattern in profanity_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError('Message contains profanity')
            
        return v
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        if v and not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('User ID can only contain alphanumeric characters, hyphens, and underscores')
        return v

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

#==================================================================================
# FastAPI App and Middleware
#==================================================================================
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

# IMPORTANT: Add the exception handler RIGHT AFTER creating the app and middleware
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"Validation handler triggered for: {request.url}")
    print(f"Validation errors: {exc.errors()}")
    
    # Extract meaningful error messages
    custom_errors = []
    error_messages = []
    
    for error in exc.errors():
        field_name = ".".join(str(loc) for loc in error["loc"][1:]) if len(error["loc"]) > 1 else "unknown"
        error_msg = error.get("msg", "Validation error")
        
        # Add to error messages list
        error_messages.append(f"{field_name}: {error_msg}")
        
        # Add to custom errors
        custom_errors.append({
            "field": field_name,
            "message": error_msg,
            "type": error.get("type", "validation_error")
        })
        
    # Return custom response
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "message": error_messages[:500],  # Ensure message isn't too long
            "errors": custom_errors,
            "detail": "Please check your input and try again."
        }
    )

# User model for authentication
class User(BaseModel):
    email: EmailStr
    name: Optional[str] = None
    picture: Optional[str] = None
    disabled: Optional[bool] = False
    password: Optional[str] = None

class UserInDB(User):
    hashed_password: str

class EmailPasswordForm(BaseModel):
    email: EmailStr
    password: str

class UserBase(BaseModel):
    email: EmailStr
    name: Optional[str] = None
    picture: Optional[str] = None
    disabled: Optional[bool] = False
    hashed_password: Optional[str] = None
    created_at: Optional[datetime] = datetime.utcnow()
    updated_at: Optional[datetime] = datetime.utcnow()

class UserCreate(UserBase):
    password: Optional[str] = None

class UserInDB(UserBase):
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

# In-memory user store (replace with database in production)
# will be replaced by mongodb
fake_users_db = {}

# MongoDB connection setup
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = "users_db"

# For async operations (recommended)
async_client = AsyncIOMotorClient(MONGODB_URI)
async_db = async_client[DB_NAME]

async def get_user_by_email(email: str) -> Optional[UserInDB]:
    """Get user by email from MongoDB"""
    try:
        user_data = await async_db.users.find_one({"email": email})
        if user_data:
            return UserInDB(**user_data)
        return None
    except PyMongoError as e:
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error")

async def create_user(user: UserCreate) -> UserInDB:
    """Create new user in MongoDB"""
    try:
        # Hash password if provided
        if user.password:
            hashed_password = pwd_context.hash(user.password)
            user_dict = user.dict(exclude={"password"})
            user_dict["hashed_password"] = hashed_password
        else:
            user_dict = user.dict()
        
        # Set timestamps
        now = datetime.utcnow()
        user_dict["created_at"] = now
        user_dict["updated_at"] = now
        
        result = await async_db.users.insert_one(user_dict)
        if result.inserted_id:
            return UserInDB(**user_dict)
        raise HTTPException(status_code=500, detail="Failed to create user")
    except PyMongoError as e:
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error")

async def update_user(email: str, update_data: dict) -> Optional[UserInDB]:
    """Update user in MongoDB"""
    try:
        update_data["updated_at"] = datetime.utcnow()
        result = await async_db.users.update_one(
            {"email": email},
            {"$set": update_data}
        )
        if result.modified_count == 1:
            return await get_user_by_email(email)
        return None
    except PyMongoError as e:
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("email")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    
    user = await get_user_by_email(token_data.email)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

@app.on_event("startup")
async def startup_db_client():
    # Create indexes
    try:
        await async_db.users.create_index("email", unique=True)
        await async_db.users.create_index("created_at")
        print("✅ MongoDB indexes created")
    except Exception as e:
        print(f"❌ Error creating indexes: {e}")

#=================================================================================
# Task Manager and Google OAuth Setup (No changes here)
#=================================================================================
# Initialize task manager
task_manager = TaskManager3()

# Add session middleware (required for Authlib)
app.add_middleware(SessionMiddleware, secret_key="your-secret-key-here")

# Initialize OAuth
oauth = OAuth()

# Register Google OAuth with correct configuration URL
oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',  # Note: openid-configuration (with hyphen)
    client_kwargs={
        'scope': 'openid email profile'  # You can use short form with Authlib
    }
)

@app.get("/auth/google")
async def login_google(request: Request):
    # Generate redirect URI dynamically
    redirect_uri = request.url_for('auth_google_callback')
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth/google/callback")
async def auth_google_callback(request: Request):
    try:
        # Get the token from Google
        token = await oauth.google.authorize_access_token(request)

        # Get user info (this is automatically included with openid scope)
        user_info = token.get('userinfo')
        
        if not user_info:
            raise HTTPException(status_code=400, detail="Failed to get user info")
        
        print(f"User info: {user_info}")  # Debug
        
        # Create or update user in your database
        email = user_info.get("email")
        if not email:
            raise HTTPException(status_code=400, detail="No email provided")
        
        # Check if user exists in MongoDB
        user = await get_user_by_email(email)
        if not user:
            # Create new user in MongoDB
            new_user = UserCreate(
                email=email,
                name=user_info.get("name"),
                picture=user_info.get("picture"),
                disabled=False
            )
            user = await create_user(new_user)
        
        # Create JWT token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"name": user.name, "email": user.email, "picture": user.picture}, 
            expires_delta=access_token_expires
        )
        
        # Redirect to frontend with token
        frontend_url = "http://localhost:3001"
        response = RedirectResponse(
            url=f"{frontend_url}/?access_token={access_token}&token_type=bearer"
        )
        return response
        
    except Exception as e:
        print(f"OAuth callback error: {str(e)}")
        raise HTTPException(
            status_code=400, 
            detail=f"Authentication failed: {str(e)}"
        )

@app.post("/register")
async def register_user(user: UserCreate):  # Changed from User to UserCreate
    """
    Register a new user (alternative to OAuth)
    """
    # Check if user exists in MongoDB
    existing_user = await get_user_by_email(user.email)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user in MongoDB
    try:
        created_user = await create_user(user)
        return {
            "message": "User created successfully", 
            "email": created_user.email,
            "name": created_user.name
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error creating user: {str(e)}"
        )

@app.post("/token", response_model=Token)
async def login_for_access_token(email: str = Form(...), password: str = Form(...)):
    """
    OAuth2 password flow for token generation (alternative to Google OAuth)
    """
    # Find user in MongoDB
    user = await get_user_by_email(email)  # username is the email in this flow
    print(user)
    print(email)
    print(password)
    if not user or not user.hashed_password or not pwd_context.verify(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"email": user.email, "name": user.name},  # Include name if available
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

#==================================================================================
# API Endpoints (All user-specific endpoints now have a security dependency)
#==================================================================================
@app.get("/")
async def root():
    return {"message": "Task Management API", "endpoints": ["/work/tasks", "/personal/tasks", "/work/summary", "/personal/summary"]}

# Work Assistant Endpoints
@app.post("/work/tasks", response_model=TaskResponse)
async def create_work_tasks(request: TaskRequest, current_user: User = Depends(get_current_active_user)):
    try:
        work_assistant = task_manager.get_assistant('work', user_id=current_user.email)
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
async def get_work_tasks(current_user: User = Depends(get_current_active_user)):
    try:
        work_assistant = task_manager.get_assistant('work', user_id=current_user.email)
        if not work_assistant:
            raise HTTPException(status_code=500, detail="Work assistant not available")
        
        tasks = work_assistant.get_all_tasks()
        return TaskListResponse(success=True, tasks=tasks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving work tasks: {str(e)}")

@app.post("/work/summary", response_model=TaskSummaryResponse)
async def get_work_summary(request: TaskRequest, current_user: User = Depends(get_current_active_user)):
    try:
        work_assistant = task_manager.get_assistant('work', user_id=current_user.email)
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
async def create_personal_tasks(request: TaskRequest, current_user: User = Depends(get_current_active_user)):
    try:
        personal_assistant = task_manager.get_assistant('personal', user_id=current_user.email)
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
async def get_personal_tasks(current_user: User = Depends(get_current_active_user)):
    try:
        personal_assistant = task_manager.get_assistant('personal', user_id=current_user.email)
        if not personal_assistant:
            raise HTTPException(status_code=500, detail="Personal assistant not available")
        
        tasks = personal_assistant.get_all_tasks()
        return TaskListResponse(success=True, tasks=tasks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving personal tasks: {str(e)}")

@app.post("/personal/summary", response_model=TaskSummaryResponse)
async def get_personal_summary(request: TaskRequest, current_user: User = Depends(get_current_active_user)):
    try:
        personal_assistant = task_manager.get_assistant('personal', user_id=current_user.email)
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
async def complete_work_task(task_id: str, current_user: User = Depends(get_current_active_user)):
    try:
        work_assistant = task_manager.get_assistant('work', user_id=current_user.email)
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
async def complete_personal_task(task_id: str, current_user: User = Depends(get_current_active_user)):
    try:
        personal_assistant = task_manager.get_assistant('personal', user_id=current_user.email)
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

# Task delete endpoints
@app.put("/work/tasks/{task_id}/delete")
async def delete_work_task(task_id: str, current_user: User = Depends(get_current_active_user)):
    try:
        work_assistant = task_manager.get_assistant('work', user_id=current_user.email)
        if not work_assistant:
            raise HTTPException(status_code=500, detail="Work assistant not available")
        
        result = work_assistant.delete_task(task_id)
        if not result:
            raise HTTPException(status_code=404, detail="Delete not successful")
        
        return {"success": True, "message": f"Task {task_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting task: {str(e)}")

@app.put("/personal/tasks/{task_id}/delete")
async def delete_personal_task(task_id: str, current_user: User = Depends(get_current_active_user)):
    try:
        personal_assistant = task_manager.get_assistant('personal', user_id=current_user.email)
        if not personal_assistant:
            raise HTTPException(status_code=500, detail="Personal assistant not available")
        
        result = personal_assistant.delete_task(task_id)
        if not result:
            raise HTTPException(status_code=404, detail="Delete not successful")
        
        return {"success": True, "message": f"Task {task_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting task: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        # Test database connection
        conn = psycopg2.connect(os.getenv("SUPABASE_DB_CONNECTION_STRING"))
        conn.close()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": db_status,
        "active_users": len(task_manager.assistants),
        "assistant_instances": {
            user_id: list(assistants.keys()) 
            for user_id, assistants in task_manager.assistants.items()
        }
    }

# Add this simple test endpoint to verify the exception handler works
@app.post("/test-validation")
async def test_validation(request: TaskRequest):
    return {"message": "Validation passed", "data": request.dict()}

if __name__ == "__main__":
    # For development - use uvicorn in production
    uvicorn.run(
        "task_api:app",  # Replace with your filename if different
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

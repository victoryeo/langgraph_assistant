// server side
import fs from 'fs';
import path from 'path';

const usersFile = path.join(process.cwd(), 'data', 'users.json');

export async function POST(request) {
  try {
    const { name, email, password } = await request.json()

    let users = [];
  
    try {
      const data = fs.readFileSync(usersFile, 'utf8');
      users = JSON.parse(data);
    } catch (error) {
      // File doesn't exist
      console.error("File doesn't exist")
    }

    if (!name || !email || !password) {
      return Response.json({ message: 'Missing required fields' }, { status: 400 })
    }

    if (password.length < 6) {
      return Response.json({ message: 'Password must be at least 6 characters' }, { status: 400 })
    }

    // Check if user already exists
    const existingUser = users.find(user => user.email === email)
    if (existingUser) {
      return Response.json({ message: 'User already exists' }, { status: 400 })
    }

    const newUser = {
      id: Date.now().toString(),
      name,
      email,
      password: password,
    }
    
    users.push(newUser)
    
    fs.mkdirSync(path.dirname(usersFile), { recursive: true });
    fs.writeFileSync(usersFile, JSON.stringify(users, null, 2));

    return Response.json({ message: 'User created successfully' }, { status: 201 })
  } catch (error) {
    return Response.json({ message: 'Internal server error' }, { status: 500 })
  }
}

export async function GET(request) {
  try {
    // Get all user
    const data = fs.readFileSync(usersFile, 'utf8');
    return Response.json(JSON.parse(data));
  } catch (error) {
    return Response.json({ message: 'Internal server error' }, { status: 500 })
  }
}
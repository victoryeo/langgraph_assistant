// app/api/users/route.js
import fs from 'fs';
import path from 'path';

const usersFile = path.join(process.cwd(), 'data', 'users.json');

export async function GET() {
  try {
    const data = fs.readFileSync(usersFile, 'utf8');
    return Response.json(JSON.parse(data));
  } catch (error) {
    return Response.json([]);
  }
}

export async function POST(request) {
  const newUser = await request.json();
  
  let users = [];
  try {
    const data = fs.readFileSync(usersFile, 'utf8');
    users = JSON.parse(data);
  } catch (error) {
    // File doesn't exist yet
  }
  
  users.push(newUser);
  
  // Ensure directory exists
  fs.mkdirSync(path.dirname(usersFile), { recursive: true });
  fs.writeFileSync(usersFile, JSON.stringify(users, null, 2));
  
  return Response.json(newUser);
}
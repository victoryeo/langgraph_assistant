// src/app/api/users/route.ts
import fs from 'fs';
import path from 'path';
import { NextRequest } from 'next/server';

const usersFile = path.join(process.cwd(), 'data', 'users.json');

const getUsersFromFile = () => {
  try {
    const data = fs.readFileSync(usersFile, 'utf8');
    return JSON.parse(data);
  } catch (error) {
    return [];
  }
};

const saveUsersToFile = (users: any[]) => {
  fs.mkdirSync(path.dirname(usersFile), { recursive: true });
  fs.writeFileSync(usersFile, JSON.stringify(users, null, 2));
};

export async function POST(request: NextRequest) {
  const newUser = await request.json();
  console.log(newUser)
  const users = getUsersFromFile();
  
  if (users.find((u: any) => u.email === newUser.email)) {
    console.log("User already exists")
    return Response.json({ error: 'User already exists' }, { status: 400 });
  }
  
  const userToAdd = {
    ...newUser,
    id: Date.now().toString()
  };
  console.log(userToAdd)

  users.push(userToAdd);
  saveUsersToFile(users);
  console.log("userToAdd saved")
  return Response.json(userToAdd);
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
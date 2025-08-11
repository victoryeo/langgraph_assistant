// lib/users-server.js
import 'server-only';
import fs from 'fs';
import path from 'path';

const usersFile = path.join(process.cwd(), 'data', 'users.json');

export const getUsersFromFile = () => {
  try {
    const data = fs.readFileSync(usersFile, 'utf8');
    return JSON.parse(data);
  } catch (error) {
    return [];
  }
};

export const saveUsersToFile = (users) => {
  fs.mkdirSync(path.dirname(usersFile), { recursive: true });
  fs.writeFileSync(usersFile, JSON.stringify(users, null, 2));
};
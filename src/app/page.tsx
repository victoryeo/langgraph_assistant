'use client';

import { useState } from 'react';
import WorkAssistant from './components/WorkAssistant';
import PersonalAssistant from './components/PersonalAssistant';

type AssistantType = 'work' | 'personal' | null;

export default function Home() {
  const [activeAssistant, setActiveAssistant] = useState<AssistantType>(null);

  const assistants = [
    { id: 'work', name: 'Work Assistant', emoji: '💼', color: 'bg-blue-100 hover:bg-blue-200' },
    { id: 'personal', name: 'Personal Assistant', emoji: '👤', color: 'bg-green-100 hover:bg-green-200' },
  ];

  const handleBack = () => {
    setActiveAssistant(null);
  };

  return (
    <div className="flex flex-1 flex-col justify-center items-center min-h-screen w-full h-full border-8 border-green-500 bg-white">
      <h1 className="text-4xl font-bold mb-16 text-center text-gray-800 bg-yellow-100">
        Welcome to Your AI Assistant
      </h1>
      {!activeAssistant ? (
        <div className="flex flex-col md:flex-row gap-12 justify-center items-center border-4 border-blue-500">
          {assistants.map((assistant) => (
            <button
              key={assistant.id}
              onClick={() => setActiveAssistant(assistant.id as AssistantType)}
              className={`p-8 rounded-2xl shadow-md transition-all duration-200 flex flex-col items-center justify-center h-64 w-72 ${assistant.color} border-2 border-transparent hover:border-blue-300 transform hover:scale-105`}
            >
              <span className="text-6xl mb-6">{assistant.emoji}</span>
              <h2 className="text-2xl font-semibold text-gray-800">{assistant.name}</h2>
              <p className="mt-2 text-gray-600">Click to start chatting</p>
            </button>
          ))}
        </div>
      ) : activeAssistant === 'work' ? (
        <WorkAssistant onBack={handleBack} />
      ) : (
        <PersonalAssistant onBack={handleBack} />
      )}
    </div>
  );
}

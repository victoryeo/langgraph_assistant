'use client';

import { useState, useRef, useEffect } from 'react';
import { chatWithAssistant } from '../../services/api';

type Message = {
  id: string;
  text: string;
  sender: 'user' | 'assistant';
  timestamp: Date;
};

type AssistantProps = {
  onBack: () => void;
};

export default function WorkAssistant({ onBack }: AssistantProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim()) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');

    try {
      // Call the API to get assistant's response
      const response = await chatWithAssistant({
        message: inputValue,
        assistant_type: 'work',
      });

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: response.response,
        sender: 'assistant',
        timestamp: new Date(response.timestamp),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error getting response from assistant:', error);
      
      // Show error message to user
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: 'Sorry, I encountered an error. Please try again.',
        sender: 'assistant',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto bg-white rounded-xl shadow-lg overflow-hidden flex flex-col h-[calc(100vh-4rem)] my-4">
      {/* Header */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center">
          <button 
            onClick={onBack}
            className="mr-4 p-2 rounded-full hover:bg-gray-100"
            aria-label="Back to assistant selection"
          >
            ←
          </button>
          <div>
            <h2 className="text-2xl font-semibold text-gray-800">Work Assistant</h2>
            <p className="text-sm text-gray-500">How can I help you today?</p>
          </div>
        </div>
      </div>
      
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center text-gray-400">
              <p className="text-sm mt-1">Ask me anything about work</p>
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] p-4 rounded-2xl ${
                  message.sender === 'user'
                    ? 'bg-blue-500 text-white rounded-br-none'
                    : 'bg-gray-100 text-gray-800 rounded-bl-none'
                }`}
              >
                <p className="whitespace-pre-wrap">{message.text}  - {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</p>
              </div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>
      
      {/* Input */}
      <div className="p-4 border-t border-gray-200 bg-white">
        <form onSubmit={handleSendMessage} className="flex items-center gap-2">
          <div className="flex-1 relative">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Type your message..."
              className="w-full p-3 pr-12 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <button
              type="submit"
              disabled={!inputValue.trim()}
              className={`absolute right-1 top-1/2 -translate-y-1/2 px-3 py-2 rounded-lg transition-colors flex items-center gap-1 ${inputValue.trim() ? 'text-white bg-blue-500 hover:bg-blue-600' : 'text-gray-400 bg-gray-100'}`}
              aria-label="Send message"
            >
              <span className="text-sm font-medium">Send</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

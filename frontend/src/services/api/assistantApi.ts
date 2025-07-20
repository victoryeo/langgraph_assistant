import { apiRequest } from './client';

export interface ChatMessage {
  id: string;
  text: string;
  sender: 'user' | 'assistant';
  timestamp: Date;
}

export interface ChatRequest {
  message: string;
  conversation_id?: string;
  assistant_type: 'work' | 'personal';
}

export interface ChatResponse {
  response: string;
  conversation_id: string;
  timestamp: string;
}

export const chatWithAssistant = async (data: ChatRequest): Promise<ChatResponse> => {
  const endpoint = data.assistant_type === 'work' ? '/work/tasks' : '/personal/tasks';
  
  const response = await apiRequest<any>(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message: data.message,
      user_id: 'test_user' // You might want to get this from auth context
    }),
  });

  if (response.error) {
    throw new Error(response.error);
  }

  // Map the response to match the expected ChatResponse interface
  return {
    response: response.data.assistant_response,
    conversation_id: data.conversation_id || `conv_${Date.now()}`,
    timestamp: new Date().toISOString()
  };
};

export const getChatHistory = async (conversationId: string): Promise<ChatMessage[]> => {
  const response = await apiRequest<ChatMessage[]>(`/api/chat/history/${conversationId}`, {
    method: 'GET',
  });

  if (response.error) {
    throw new Error(response.error);
  }

  // Convert string timestamps to Date objects
  return response.data!.map(msg => ({
    ...msg,
    timestamp: new Date(msg.timestamp)
  }));
};

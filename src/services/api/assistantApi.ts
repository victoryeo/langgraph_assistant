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
  const response = await apiRequest<ChatResponse>('/api/chat', {
    method: 'POST',
    body: JSON.stringify(data),
  });

  if (response.error) {
    throw new Error(response.error);
  }

  return response.data!;
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

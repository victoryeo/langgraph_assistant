import { User } from '../types/user';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Store token in memory (for SSR)
let authToken: string | null = null;

// Get token from localStorage on client side
if (typeof window !== 'undefined') {
  authToken = localStorage.getItem('authToken');
}

export const loginWithGoogle = async (): Promise<{ user: User; token: string }> => {
  try {
    // Step 1: Get the Google OAuth URL from the backend
    const response = await fetch(`${API_BASE_URL}/auth/google`);
    const { authorization_url } = await response.json();
    
    // Step 2: Redirect to Google's OAuth consent screen
    // The backend will handle the callback at /auth/google/callback
    window.location.href = authorization_url;
    
    // This will be a placeholder since we're redirecting
    return { user: {} as User, token: '' };
  } catch (error) {
    console.error('Google login error:', error);
    throw new Error('Failed to initiate Google login');
  }
};

export const handleGoogleCallback = async (code: string): Promise<{ user: User; token: string }> => {
  try {
    const response = await fetch(`${API_BASE_URL}/auth/google/callback?code=${code}`);
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      console.error('Google auth error:', errorData);
      throw new Error(errorData.detail || 'Failed to authenticate with Google');
    }
    
    const data = await response.json();
    
    // Store the token
    if (typeof window !== 'undefined') {
      localStorage.setItem('authToken', data.access_token);
      authToken = data.access_token;
    }
    
    return {
      user: data.user,
      token: data.access_token
    };
  } catch (error) {
    console.error('Google callback error:', error);
    throw new Error('Failed to complete Google authentication');
  }
};

export const getAuthToken = (): string | null => {
  return authToken || (typeof window !== 'undefined' ? localStorage.getItem('authToken') : null);
};

export const isAuthenticated = (): boolean => {
  return !!getAuthToken();
};

export const logout = (): void => {
  if (typeof window !== 'undefined') {
    localStorage.removeItem('authToken');
    authToken = null;
    // Redirect to home or login page
    window.location.href = '/';
  }
};

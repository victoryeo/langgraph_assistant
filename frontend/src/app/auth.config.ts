// server side
import Credentials from "next-auth/providers/credentials"
import type { Provider } from "next-auth/providers"
import type { NextAuthConfig } from 'next-auth';

const providers: Provider[] = [
    Credentials({
      credentials: {
        email: { label: 'Email', type: 'email' },
        password: { label: 'Password', type: 'password' }
      },
      authorize: async (credentials) => {
        console.log("authorize")
        const { getUsersFromFile } = await import('./lib/users-server');
        const users = getUsersFromFile();

        if (!credentials?.email || !credentials?.password) {
          return null
        }

        // Type assertion to ensure credentials are strings
        const email = credentials.email as string
        const password = credentials.password as string

        const user = users.find(user => user.email === email)
        console.log(users)
        console.log(email, password)
        console.log(user)
        if (!user) {
          console.error("user not found")
          return null
        }
        console.log(password, user.password)

        const isPasswordValid = (password == user.password)
        
        if (!isPasswordValid) {
          return null
        }

        return {
          id: user.id,
          email: user.email,
          name: user.name,
        }
      }
    })
  ]

export const providerMap = providers.map((provider) => {
	if (typeof provider === "function") {
		const providerData = provider()
		console.log("1", providerData.id, providerData.name)
		return { id: providerData.id, name: providerData.name }
	} else {
		console.log("2", provider.id, provider.name)
		return { id: provider.id, name: provider.name }
	}
})

export const authConfig = {
  pages: {
    signIn: '/signin',
  },
  callbacks: {
    authorized({ auth, request: { nextUrl } }) {
      return true;
    },   
  },
  providers: providers,
} satisfies NextAuthConfig;
'use server'

import { signIn } from "../auth"
import { redirect } from 'next/navigation'

export async function handleSignIn(formData) {
  try {
    const email = formData.get('email')
    const password = formData.get('password')
    console.log(email, password)
    await signIn("credentials", {
      email,
      password,
      redirectTo: "/",
    })
  } catch (error) {
    return { error: "Invalid email or password" }
  }
  
  redirect('/')
}
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'LangGraph Assistant',
  description: 'Your personal and work AI assistant',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" style={{ height: '100%' }}>
      <body className={inter.className} style={{ minHeight: '100vh', height: '100%', width: '100%' }}>{children}</body>
    </html>
  );
}

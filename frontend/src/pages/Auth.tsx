import { useState } from 'react';
import { motion } from 'framer-motion';
import { SignIn, SignUp } from '@clerk/clerk-react';

export default function Auth() {
  const [mode, setMode] = useState<'signin' | 'signup'>('signin');

  return (
    <div className="min-h-[calc(100vh-72px)] flex items-center justify-center px-6 py-12 relative overflow-hidden">
      {/* Background decorations */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-[-100px] right-[-100px] w-[500px] h-[500px] opacity-10 rounded-full"
          style={{ background: 'radial-gradient(circle, #4A90D9, transparent)' }}
        />
        <div className="absolute bottom-[-100px] left-[-100px] w-[400px] h-[400px] opacity-10 rounded-full"
          style={{ background: 'radial-gradient(circle, #8B5CF6, transparent)' }}
        />
        {/* Floating books */}
        {[
          { top: '10%', left: '5%', rotate: '-12deg', color: '#4A90D9', delay: 0 },
          { top: '60%', right: '8%', rotate: '15deg', color: '#8B5CF6', delay: 1 },
          { top: '30%', right: '15%', rotate: '-8deg', color: '#14B8A6', delay: 2 },
          { top: '70%', left: '10%', rotate: '10deg', color: '#F59E0B', delay: 0.5 },
        ].map((book, i) => (
          <motion.div
            key={i}
            animate={{ y: [0, -15, 0] }}
            transition={{ duration: 5 + i, repeat: Infinity, ease: 'easeInOut', delay: book.delay }}
            className="absolute w-12 h-16 rounded-[6px] opacity-8"
            style={{
              top: book.top,
              left: (book as any).left,
              right: (book as any).right,
              background: `linear-gradient(135deg, ${book.color}40, ${book.color}20)`,
              border: `1px solid ${book.color}20`,
              transform: `rotate(${book.rotate}) perspective(500px) rotateY(10deg)`,
            }}
          />
        ))}
      </div>

      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="relative z-10 w-full max-w-md"
      >
        {/* Card */}
        <div className="rounded-[28px] overflow-hidden"
          style={{
            background: 'linear-gradient(145deg, #ffffff, #f8f6f2)',
            boxShadow: '0 20px 60px rgba(0,0,0,0.08), 0 0 0 1px rgba(0,0,0,0.02), inset 0 2px 0 rgba(255,255,255,1)',
          }}
        >
          {/* Header */}
          <div className="p-8 pb-0 text-center">
            <motion.div
              animate={{ rotate: [0, -5, 5, 0] }}
              transition={{ duration: 3, repeat: Infinity }}
              className="w-16 h-16 rounded-[18px] flex items-center justify-center text-3xl mx-auto mb-5"
              style={{
                background: 'linear-gradient(135deg, #4A90D9, #8B5CF6)',
                boxShadow: '0 8px 25px rgba(74,144,217,0.3)',
              }}
            >
              📚
            </motion.div>
            <h1 className="text-2xl font-extrabold tracking-tight mb-1">
              {mode === 'signin' ? 'Welcome back' : 'Create account'}
            </h1>
            <p className="text-sm text-text-tertiary">
              {mode === 'signin' ? 'Sign in to continue your reading journey' : 'Join thousands of book lovers'}
            </p>
          </div>

          {/* Tab Toggle */}
          <div className="px-8 pt-6">
            <div className="flex rounded-[14px] p-1" style={{
              background: 'linear-gradient(145deg, #f0eeea, #f5f3ee)',
              boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.04)',
            }}>
              {(['signin', 'signup'] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  className={`flex-1 py-2.5 rounded-[11px] text-sm font-bold transition-all duration-300 ${
                    mode === m ? 'text-text-primary' : 'text-text-muted'
                  }`}
                  style={mode === m ? {
                    background: 'linear-gradient(145deg, #ffffff, #fafaf8)',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.06), inset 0 1px 0 rgba(255,255,255,0.9)',
                  } : {}}
                >
                  {m === 'signin' ? 'Sign In' : 'Sign Up'}
                </button>
              ))}
            </div>
          </div>

          {/* Clerk Auth Component */}
          <div className="p-6 flex justify-center">
            {mode === 'signin' ? (
              <SignIn
                routing="hash"
                appearance={{
                  elements: {
                    rootBox: 'w-full',
                    card: 'shadow-none bg-transparent w-full',
                    headerTitle: 'hidden',
                    headerSubtitle: 'hidden',
                    socialButtonsBlockButton: 'rounded-[14px] font-semibold border border-gray-100',
                    formButtonPrimary: 'rounded-[14px] bg-gradient-to-r from-[#4A90D9] to-[#8B5CF6] font-bold text-sm shadow-lg shadow-blue-200/50 hover:shadow-blue-300/50',
                    formFieldInput: 'rounded-[14px] border-gray-100 bg-[#f5f3ee] focus:ring-2 focus:ring-blue-200 focus:border-transparent',
                    formFieldLabel: 'text-xs font-bold text-[#4A4A6A] uppercase tracking-wider',
                    footerActionLink: 'text-[#4A90D9] font-semibold',
                    dividerLine: 'bg-gray-100',
                    dividerText: 'text-[#B0B0C8] text-xs font-semibold',
                    footer: 'hidden',
                  },
                }}
              />
            ) : (
              <SignUp
                routing="hash"
                appearance={{
                  elements: {
                    rootBox: 'w-full',
                    card: 'shadow-none bg-transparent w-full',
                    headerTitle: 'hidden',
                    headerSubtitle: 'hidden',
                    socialButtonsBlockButton: 'rounded-[14px] font-semibold border border-gray-100',
                    formButtonPrimary: 'rounded-[14px] bg-gradient-to-r from-[#4A90D9] to-[#8B5CF6] font-bold text-sm shadow-lg shadow-blue-200/50 hover:shadow-blue-300/50',
                    formFieldInput: 'rounded-[14px] border-gray-100 bg-[#f5f3ee] focus:ring-2 focus:ring-blue-200 focus:border-transparent',
                    formFieldLabel: 'text-xs font-bold text-[#4A4A6A] uppercase tracking-wider',
                    footerActionLink: 'text-[#4A90D9] font-semibold',
                    dividerLine: 'bg-gray-100',
                    dividerText: 'text-[#B0B0C8] text-xs font-semibold',
                    footer: 'hidden',
                  },
                }}
              />
            )}
          </div>
        </div>
      </motion.div>
    </div>
  );
}

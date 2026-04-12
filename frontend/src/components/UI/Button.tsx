import { motion } from 'framer-motion';
import type { HTMLMotionProps } from 'framer-motion';

interface ButtonProps extends HTMLMotionProps<'button'> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'glow';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  children: React.ReactNode;
}

const variants = {
  primary: {
    background: 'linear-gradient(135deg, #4A90D9, #8B5CF6)',
    color: '#fff',
    boxShadow: '0 4px 15px rgba(74,144,217,0.25), inset 0 1px 0 rgba(255,255,255,0.2)',
  },
  secondary: {
    background: 'linear-gradient(145deg, #ffffff, #f0eeea)',
    color: '#4A4A6A',
    boxShadow: '0 4px 12px rgba(0,0,0,0.06), 0 0 0 1px rgba(0,0,0,0.03), inset 0 1px 0 rgba(255,255,255,0.9)',
  },
  ghost: {
    background: 'transparent',
    color: '#4A4A6A',
    boxShadow: 'none',
  },
  glow: {
    background: 'linear-gradient(135deg, #4A90D9, #8B5CF6)',
    color: '#fff',
    boxShadow: '0 0 20px rgba(74,144,217,0.3), 0 0 40px rgba(139,92,246,0.15), inset 0 1px 0 rgba(255,255,255,0.2)',
  },
};

const sizes = {
  sm: 'px-4 py-2 text-xs rounded-[10px]',
  md: 'px-6 py-3 text-sm rounded-[14px]',
  lg: 'px-8 py-4 text-base rounded-[16px]',
};

export default function Button({
  variant = 'primary', size = 'md', loading, children, className = '', ...props
}: ButtonProps) {
  return (
    <motion.button
      whileHover={{
        y: -2,
        boxShadow: variant === 'glow'
          ? '0 0 30px rgba(74,144,217,0.4), 0 0 60px rgba(139,92,246,0.2)'
          : variant === 'primary'
          ? '0 8px 25px rgba(74,144,217,0.3), inset 0 1px 0 rgba(255,255,255,0.2)'
          : '0 8px 20px rgba(0,0,0,0.08)',
      }}
      whileTap={{ scale: 0.97, y: 0 }}
      className={`font-bold tracking-tight transition-all duration-300 inline-flex items-center justify-center gap-2 ${sizes[size]} ${className}`}
      style={variants[variant]}
      disabled={loading}
      {...props}
    >
      {loading ? (
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
          className="w-4 h-4 border-2 border-current border-t-transparent rounded-full"
        />
      ) : null}
      {children}
    </motion.button>
  );
}

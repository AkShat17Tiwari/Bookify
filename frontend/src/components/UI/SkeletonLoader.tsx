import { motion } from 'framer-motion';

interface SkeletonProps {
  variant?: 'card' | 'line' | 'circle' | 'chart';
  count?: number;
}

function SkeletonCard() {
  return (
    <div className="rounded-[22px] overflow-hidden" style={{
      background: 'linear-gradient(145deg, #ffffff, #f5f3ee)',
      boxShadow: '0 2px 8px rgba(0,0,0,0.04), inset 0 1px 0 rgba(255,255,255,0.9)',
    }}>
      <div className="aspect-[2/3] animate-shimmer rounded-t-[22px]" />
      <div className="p-5 space-y-3">
        <div className="h-4 rounded-full animate-shimmer w-3/4" />
        <div className="h-3 rounded-full animate-shimmer w-1/2" />
        <div className="flex gap-2 pt-2">
          <div className="h-3 w-3 rounded-full animate-shimmer" />
          <div className="h-3 w-3 rounded-full animate-shimmer" />
          <div className="h-3 w-3 rounded-full animate-shimmer" />
          <div className="h-3 w-3 rounded-full animate-shimmer" />
          <div className="h-3 w-3 rounded-full animate-shimmer" />
        </div>
      </div>
    </div>
  );
}

function SkeletonLine({ width = '100%' }: { width?: string }) {
  return <div className="h-4 rounded-full animate-shimmer" style={{ width }} />;
}

function SkeletonCircle({ size = 48 }: { size?: number }) {
  return <div className="rounded-full animate-shimmer" style={{ width: size, height: size }} />;
}

export default function SkeletonLoader({ variant = 'card', count = 1 }: SkeletonProps) {
  return (
    <>
      {Array.from({ length: count }, (_, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: i * 0.05 }}
        >
          {variant === 'card' && <SkeletonCard />}
          {variant === 'line' && <SkeletonLine />}
          {variant === 'circle' && <SkeletonCircle />}
          {variant === 'chart' && (
            <div className="rounded-[20px] p-6 space-y-4" style={{
              background: 'linear-gradient(145deg, #ffffff, #f5f3ee)',
              boxShadow: '0 2px 8px rgba(0,0,0,0.04)',
            }}>
              <SkeletonLine width="40%" />
              <div className="h-48 rounded-[14px] animate-shimmer" />
            </div>
          )}
        </motion.div>
      ))}
    </>
  );
}

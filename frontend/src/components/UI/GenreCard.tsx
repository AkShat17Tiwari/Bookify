import { motion } from 'framer-motion';
import { fadeUp } from '../../lib/animations';

const genreConfig: Record<string, { emoji: string; gradient: string; shadow: string }> = {
  'Science Fiction': { emoji: '🚀', gradient: 'from-blue-400/20 to-indigo-400/20', shadow: 'rgba(59,130,246,0.1)' },
  'Romance': { emoji: '💕', gradient: 'from-pink-400/20 to-rose-400/20', shadow: 'rgba(244,63,94,0.1)' },
  'Mystery/Thriller': { emoji: '🔍', gradient: 'from-amber-400/20 to-orange-400/20', shadow: 'rgba(245,158,11,0.1)' },
  'Horror': { emoji: '👻', gradient: 'from-purple-400/20 to-violet-400/20', shadow: 'rgba(139,92,246,0.1)' },
  'Fantasy': { emoji: '🧙', gradient: 'from-emerald-400/20 to-teal-400/20', shadow: 'rgba(20,184,166,0.1)' },
  'Literary Fiction': { emoji: '📖', gradient: 'from-cyan-400/20 to-sky-400/20', shadow: 'rgba(14,165,233,0.1)' },
  'Non-Fiction': { emoji: '📰', gradient: 'from-slate-400/20 to-gray-400/20', shadow: 'rgba(100,116,139,0.1)' },
  'Biography': { emoji: '👤', gradient: 'from-amber-300/20 to-yellow-400/20', shadow: 'rgba(245,158,11,0.1)' },
  'Self-Help': { emoji: '💪', gradient: 'from-green-400/20 to-emerald-400/20', shadow: 'rgba(16,185,129,0.1)' },
  'History': { emoji: '🏛️', gradient: 'from-orange-400/20 to-red-400/20', shadow: 'rgba(249,115,22,0.1)' },
  'Classics': { emoji: '📚', gradient: 'from-yellow-400/20 to-amber-400/20', shadow: 'rgba(245,158,11,0.1)' },
  'Poetry': { emoji: '✒️', gradient: 'from-violet-400/20 to-purple-400/20', shadow: 'rgba(139,92,246,0.1)' },
  'Young Adult': { emoji: '🌟', gradient: 'from-pink-300/20 to-purple-300/20', shadow: 'rgba(168,85,247,0.1)' },
  'Children': { emoji: '🧸', gradient: 'from-sky-300/20 to-blue-300/20', shadow: 'rgba(56,189,248,0.1)' },
  'Cooking': { emoji: '🍳', gradient: 'from-red-400/20 to-orange-400/20', shadow: 'rgba(239,68,68,0.1)' },
  'Travel': { emoji: '✈️', gradient: 'from-teal-400/20 to-cyan-400/20', shadow: 'rgba(20,184,166,0.1)' },
  'Fiction': { emoji: '📕', gradient: 'from-indigo-400/20 to-blue-400/20', shadow: 'rgba(99,102,241,0.1)' },
  'Religious/Spiritual': { emoji: '🕊️', gradient: 'from-yellow-300/20 to-amber-300/20', shadow: 'rgba(252,211,77,0.1)' },
};

interface GenreCardProps {
  genre: string;
  count?: number;
  index?: number;
  onClick?: () => void;
}

export default function GenreCard({ genre, count, index = 0, onClick }: GenreCardProps) {
  const config = genreConfig[genre] || { emoji: '📚', gradient: 'from-gray-400/20 to-slate-400/20', shadow: 'rgba(0,0,0,0.05)' };

  return (
    <motion.div
      variants={fadeUp}
      custom={index}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true }}
      whileHover={{ y: -6, scale: 1.03 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      className={`cursor-pointer rounded-[18px] p-5 bg-gradient-to-br ${config.gradient} relative overflow-hidden group`}
      style={{
        boxShadow: `0 4px 15px ${config.shadow}, inset 0 1px 0 rgba(255,255,255,0.6)`,
        border: '1px solid rgba(255,255,255,0.5)',
      }}
    >
      {/* Hover glow */}
      <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-[18px]"
        style={{ boxShadow: `inset 0 0 30px ${config.shadow}` }}
      />

      <div className="text-3xl mb-3">{config.emoji}</div>
      <h3 className="text-sm font-bold text-text-primary leading-tight">{genre}</h3>
      {count !== undefined && (
        <p className="text-xs font-semibold text-text-muted mt-1">{count.toLocaleString()} books</p>
      )}
    </motion.div>
  );
}

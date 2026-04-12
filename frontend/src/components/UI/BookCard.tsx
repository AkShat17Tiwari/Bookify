import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { useTilt } from '../../hooks/useTilt';
import { fadeUp } from '../../lib/animations';
import { HiHeart, HiOutlineHeart, HiOutlineStar, HiStar } from 'react-icons/hi';

interface BookCardProps {
  title: string;
  author: string;
  image: string;
  rating?: number;
  votes?: number;
  reasons?: string[];
  index?: number;
  onWishlistToggle?: (title: string) => void;
  isWishlisted?: boolean;
  onClick?: () => void;
}

export default function BookCard({
  title, author, image, rating, votes, reasons, index = 0,
  onWishlistToggle, isWishlisted = false, onClick,
}: BookCardProps) {
  const { ref, style, onMouseMove, onMouseLeave } = useTilt(6);
  const [imgError, setImgError] = useState(false);
  const [heartAnimating, setHeartAnimating] = useState(false);

  const handleWishlist = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    setHeartAnimating(true);
    setTimeout(() => setHeartAnimating(false), 500);
    onWishlistToggle?.(title);
  }, [onWishlistToggle, title]);

  const starCount = rating ? Math.round(rating) : 0;

  const fallbackImage = `https://via.placeholder.com/200x300/E8F0FE/4A90D9?text=${encodeURIComponent(title.substring(0, 20))}`;

  return (
    <motion.div
      ref={ref}
      variants={fadeUp}
      custom={index}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, margin: '-50px' }}
      onMouseMove={onMouseMove}
      onMouseLeave={onMouseLeave}
      onClick={onClick}
      className="group cursor-pointer"
      style={{ perspective: 800 }}
    >
      <motion.div
        animate={{ rotateX: style.rotateX, rotateY: style.rotateY }}
        transition={{ duration: 0.1, ease: 'linear' }}
        whileHover={{
          y: -12,
          transition: { duration: 0.3, ease: [0.25, 0.46, 0.45, 0.94] },
        }}
        className="relative rounded-[22px] overflow-hidden paper-texture"
        style={{
          background: 'linear-gradient(145deg, #ffffff, #f5f3ee)',
          boxShadow: `
            0 4px 6px -1px rgba(0, 0, 0, 0.05),
            0 2px 4px -2px rgba(0, 0, 0, 0.03),
            0 0 0 1px rgba(0, 0, 0, 0.02),
            inset 0 1px 0 rgba(255, 255, 255, 0.9)
          `,
          transformStyle: 'preserve-3d',
        }}
      >
        {/* Hover glow border */}
        <div className="absolute inset-0 rounded-[22px] opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"
          style={{
            background: 'linear-gradient(135deg, rgba(74,144,217,0.1), rgba(139,92,246,0.08))',
            border: '1px solid rgba(74,144,217,0.15)',
          }}
        />

        {/* Top accent line */}
        <div className="absolute top-0 left-0 right-0 h-[3px] opacity-0 group-hover:opacity-100 transition-opacity duration-500"
          style={{
            background: 'linear-gradient(90deg, #4A90D9, #8B5CF6, #F43F5E)',
            backgroundSize: '200% 100%',
            animation: 'gradient-shift 3s linear infinite',
          }}
        />

        {/* Wishlist Button */}
        <motion.button
          onClick={handleWishlist}
          animate={heartAnimating ? { scale: [1, 1.3, 0.95, 1.15, 1] } : {}}
          transition={{ duration: 0.5 }}
          className="absolute top-3 right-3 z-10 w-9 h-9 rounded-[12px] flex items-center justify-center transition-all duration-300"
          style={{
            background: 'rgba(255,255,255,0.85)',
            backdropFilter: 'blur(8px)',
            boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
          }}
        >
          {isWishlisted ? (
            <HiHeart className="text-lg text-bookify-rose" />
          ) : (
            <HiOutlineHeart className="text-lg text-text-tertiary group-hover:text-bookify-rose transition-colors" />
          )}
        </motion.button>

        {/* Cover Image */}
        <div className="relative w-full aspect-[2/3] overflow-hidden bg-pastel-blue">
          <img
            src={imgError ? fallbackImage : image}
            alt={title}
            loading="lazy"
            onError={() => setImgError(true)}
            className="w-full h-full object-cover transition-transform duration-700 ease-out group-hover:scale-[1.06]"
          />
          {/* Bottom gradient overlay */}
          <div className="absolute bottom-0 left-0 right-0 h-24 bg-gradient-to-t from-white/90 to-transparent" />
        </div>

        {/* Content */}
        <div className="p-5 pt-3">
          <h3 className="text-[15px] font-bold text-text-primary leading-tight mb-1.5 line-clamp-2 group-hover:text-bookify-blue transition-colors">
            {title}
          </h3>
          <p className="text-[13px] font-medium text-bookify-purple/70 mb-3">
            {author}
          </p>

          {/* Rating */}
          {rating !== undefined && (
            <div className="flex items-center justify-between pt-3 border-t border-surface-3/50">
              <div className="flex items-center gap-1">
                {Array.from({ length: 5 }, (_, i) => (
                  <motion.span
                    key={i}
                    initial={{ opacity: 0, scale: 0 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: i * 0.08, duration: 0.3 }}
                  >
                    {i < starCount ? (
                      <HiStar className="text-bookify-amber text-sm" />
                    ) : (
                      <HiOutlineStar className="text-cream-400 text-sm" />
                    )}
                  </motion.span>
                ))}
                <span className="text-xs font-bold text-bookify-amber ml-1">
                  {rating?.toFixed(1)}
                </span>
              </div>
              {votes !== undefined && (
                <span className="text-[11px] font-semibold text-text-muted">
                  {votes.toLocaleString()} votes
                </span>
              )}
            </div>
          )}

          {/* AI Reasons */}
          {reasons && reasons.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1.5">
              {reasons.slice(0, 2).map((reason, i) => (
                <span key={i} className="text-[10px] font-semibold px-2.5 py-1 rounded-full bg-pastel-blue text-bookify-blue">
                  {reason}
                </span>
              ))}
            </div>
          )}
        </div>
      </motion.div>
    </motion.div>
  );
}

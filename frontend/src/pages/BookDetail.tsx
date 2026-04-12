import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { staggerContainer, fadeUp } from '../lib/animations';
import { useApi } from '../lib/api';
import type { BookDetails } from '../lib/api';
import Button from '../components/UI/Button';
import { HiHeart, HiOutlineHeart, HiStar, HiOutlineStar, HiOutlineBookOpen, HiOutlineClock, HiOutlineTag } from 'react-icons/hi';

export default function BookDetail() {
  const { title } = useParams<{ title: string }>();
  const navigate = useNavigate();
  const bookTitle = decodeURIComponent(title || '');
  const api = useApi();

  const [details, setDetails] = useState<BookDetails | null>(null);
  const [loading, setLoading] = useState(true);
  const [wishlisted, setWishlisted] = useState(false);
  const [userRating, setUserRating] = useState<number>(0);
  const [hoverRating, setHoverRating] = useState(0);
  const [markedRead, setMarkedRead] = useState(false);

  useEffect(() => {
    if (!bookTitle) return;
    const load = async () => {
      setLoading(true);
      try {
        const [detailsData, wishCheck, ratingCheck] = await Promise.allSettled([
          api.getBookDetails(bookTitle),
          api.checkWishlist(bookTitle),
          api.checkRating(bookTitle),
        ]);
        if (detailsData.status === 'fulfilled') setDetails(detailsData.value);
        if (wishCheck.status === 'fulfilled') setWishlisted(wishCheck.value.wishlisted);
        if (ratingCheck.status === 'fulfilled' && ratingCheck.value.rating) setUserRating(ratingCheck.value.rating);
      } catch { }
      setLoading(false);
    };
    load();
  }, [bookTitle]);

  const toggleWishlist = async () => {
    try {
      if (wishlisted) {
        await api.removeFromWishlist(bookTitle);
      } else {
        await api.addToWishlist(bookTitle, '', '');
      }
      setWishlisted(!wishlisted);
    } catch { }
  };

  const handleRate = async (rating: number) => {
    setUserRating(rating);
    try {
      await api.rateBook(bookTitle, rating);
    } catch { }
  };

  const handleMarkRead = async () => {
    setMarkedRead(!markedRead);
    try {
      if (!markedRead) {
        await api.addToHistory(bookTitle, '', '');
      } else {
        await api.removeFromHistory(bookTitle);
      }
    } catch { }
  };

  return (
    <div className="min-h-screen px-6 py-12">
      <div className="max-w-4xl mx-auto">
        <motion.button
          variants={fadeUp}
          initial="hidden"
          animate="visible"
          onClick={() => navigate(-1)}
          className="text-sm font-semibold text-text-tertiary hover:text-text-primary mb-8 flex items-center gap-1"
        >
          ← Back
        </motion.button>

        <motion.div
          variants={staggerContainer}
          initial="hidden"
          animate="visible"
          className="rounded-[28px] overflow-hidden"
          style={{
            background: 'linear-gradient(145deg, #ffffff, #f8f6f2)',
            boxShadow: '0 15px 50px rgba(0,0,0,0.06), 0 0 0 1px rgba(0,0,0,0.02), inset 0 2px 0 rgba(255,255,255,1)',
          }}
        >
          {/* Hero area */}
          <div className="relative">
            {/* Blurred backdrop */}
            <div className="absolute inset-0 h-64 overflow-hidden">
              <div className="w-full h-full opacity-30 blur-3xl scale-110"
                style={{ background: 'linear-gradient(135deg, #4A90D9, #8B5CF6, #F43F5E)' }}
              />
            </div>

            <div className="relative flex flex-col md:flex-row gap-8 p-8 md:p-12">
              {/* Book Cover */}
              <motion.div
                variants={fadeUp}
                custom={0}
                className="flex-shrink-0 mx-auto md:mx-0"
              >
                <div className="w-48 h-72 rounded-[18px] overflow-hidden"
                  style={{
                    boxShadow: '0 15px 40px rgba(0,0,0,0.15), 0 5px 15px rgba(0,0,0,0.08)',
                  }}
                >
                  <img
                    src={`https://covers.openlibrary.org/b/title/${encodeURIComponent(bookTitle)}-M.jpg`}
                    alt={bookTitle}
                    className="w-full h-full object-cover"
                    onError={(e) => {
                      (e.target as HTMLImageElement).src = `https://via.placeholder.com/200x300/E8F0FE/4A90D9?text=${encodeURIComponent(bookTitle.substring(0, 15))}`;
                    }}
                  />
                </div>
              </motion.div>

              {/* Info */}
              <div className="flex-1">
                <motion.h1 variants={fadeUp} custom={1} className="text-2xl md:text-3xl font-extrabold text-text-primary tracking-tight mb-3">
                  {bookTitle}
                </motion.h1>

                {/* Meta pills */}
                <motion.div variants={fadeUp} custom={2} className="flex flex-wrap gap-2 mb-6">
                  {details?.year && (
                    <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold"
                      style={{ background: 'rgba(74,144,217,0.08)', color: '#4A90D9', border: '1px solid rgba(74,144,217,0.12)' }}
                    >
                      <HiOutlineClock /> {details.year}
                    </span>
                  )}
                  {details?.pages && (
                    <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold"
                      style={{ background: 'rgba(139,92,246,0.08)', color: '#8B5CF6', border: '1px solid rgba(139,92,246,0.12)' }}
                    >
                      <HiOutlineBookOpen /> {details.pages} pages
                    </span>
                  )}
                  {details?.subjects?.slice(0, 3).map((s) => (
                    <span key={s} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold"
                      style={{ background: 'rgba(20,184,166,0.08)', color: '#14B8A6', border: '1px solid rgba(20,184,166,0.12)' }}
                    >
                      <HiOutlineTag /> {s}
                    </span>
                  ))}
                </motion.div>

                {/* User Rating */}
                <motion.div variants={fadeUp} custom={3} className="mb-6">
                  <span className="text-xs font-bold text-text-muted uppercase tracking-wider block mb-2">Your Rating</span>
                  <div className="flex items-center gap-1.5">
                    {Array.from({ length: 5 }, (_, i) => (
                      <motion.button
                        key={i}
                        whileHover={{ scale: 1.2 }}
                        whileTap={{ scale: 0.9 }}
                        onMouseEnter={() => setHoverRating(i + 1)}
                        onMouseLeave={() => setHoverRating(0)}
                        onClick={() => handleRate(i + 1)}
                        className="text-2xl transition-colors"
                      >
                        {i < (hoverRating || userRating) ? (
                          <HiStar className="text-bookify-amber" />
                        ) : (
                          <HiOutlineStar className="text-cream-400" />
                        )}
                      </motion.button>
                    ))}
                    {userRating > 0 && (
                      <span className="text-sm font-bold text-bookify-amber ml-2">{userRating}/5</span>
                    )}
                  </div>
                </motion.div>

                {/* Action Buttons */}
                <motion.div variants={fadeUp} custom={4} className="flex flex-wrap gap-3">
                  <Button
                    variant={wishlisted ? 'primary' : 'secondary'}
                    onClick={toggleWishlist}
                  >
                    {wishlisted ? <HiHeart /> : <HiOutlineHeart />}
                    {wishlisted ? 'Wishlisted' : 'Add to Wishlist'}
                  </Button>
                  <Button
                    variant={markedRead ? 'primary' : 'secondary'}
                    onClick={handleMarkRead}
                  >
                    <HiOutlineBookOpen />
                    {markedRead ? 'Read ✓' : 'Mark as Read'}
                  </Button>
                </motion.div>
              </div>
            </div>
          </div>

          {/* Description */}
          <div className="px-8 md:px-12 pb-10">
            {loading ? (
              <div className="space-y-3">
                <div className="h-4 rounded-full animate-shimmer w-3/4" />
                <div className="h-4 rounded-full animate-shimmer w-full" />
                <div className="h-4 rounded-full animate-shimmer w-5/6" />
              </div>
            ) : details?.description ? (
              <motion.div variants={fadeUp} initial="hidden" animate="visible">
                <h3 className="text-sm font-bold text-text-primary uppercase tracking-wider mb-3">About This Book</h3>
                <p className="text-text-secondary leading-relaxed">{details.description}</p>
              </motion.div>
            ) : (
              <p className="text-sm text-text-muted italic">No description available from Open Library.</p>
            )}
          </div>
        </motion.div>
      </div>
    </div>
  );
}

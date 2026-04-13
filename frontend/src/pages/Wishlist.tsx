import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { staggerContainer, fadeUp } from '../lib/animations';
import BookCard from '../components/UI/BookCard';
import SkeletonLoader from '../components/UI/SkeletonLoader';
import Button from '../components/UI/Button';
import { useApi } from '../lib/api';
import type { WishlistItem } from '../lib/api';
import { HiOutlineHeart, HiHeart } from 'react-icons/hi';

export default function Wishlist() {
  const navigate = useNavigate();
  const api = useApi();
  const [books, setBooks] = useState<WishlistItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadWishlist();
  }, []);

  const loadWishlist = async () => {
    setLoading(true);
    try {
      const data = await api.getWishlist();
      setBooks(data.wishlist || []);
    } catch (err) {
      console.error('Failed to load wishlist:', err);
      setBooks([]);
    }
    setLoading(false);
  };

  const removeBook = async (title: string) => {
    setBooks((prev) => prev.filter((b) => b.book_title !== title));
    try {
      await api.removeFromWishlist(title);
    } catch { }
  };

  return (
    <div className="min-h-screen px-6 py-12">
      <div className="max-w-[1400px] mx-auto">
        {/* Header */}
        <motion.div
          variants={staggerContainer}
          initial="hidden"
          animate="visible"
          className="mb-10"
        >
          <motion.div variants={fadeUp} custom={0} className="flex items-center gap-3 mb-2">
            <div className="w-12 h-12 rounded-[16px] flex items-center justify-center"
              style={{
                background: 'linear-gradient(135deg, rgba(244,63,94,0.1), rgba(244,63,94,0.05))',
                border: '1px solid rgba(244,63,94,0.15)',
              }}
            >
              <HiHeart className="text-xl text-bookify-rose" />
            </div>
            <div>
              <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight">My Wishlist</h1>
              <p className="text-sm text-text-tertiary">{books.length} books saved for later</p>
            </div>
          </motion.div>
        </motion.div>

        {/* Loading */}
        {loading && (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6">
            <SkeletonLoader variant="card" count={6} />
          </div>
        )}

        {/* Empty State */}
        {!loading && books.length === 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center py-24"
          >
            <motion.div
              animate={{ y: [0, -10, 0] }}
              transition={{ duration: 3, repeat: Infinity }}
              className="text-6xl mb-6"
            >
              💕
            </motion.div>
            <h2 className="text-xl font-bold text-text-primary mb-3">Your wishlist is empty</h2>
            <p className="text-text-tertiary mb-6 max-w-md mx-auto">
              Start exploring books and save your favorites here for later reading.
            </p>
            <Button variant="primary" onClick={() => navigate('/search')}>
              <HiOutlineHeart /> Discover Books
            </Button>
          </motion.div>
        )}

        {/* Wishlist Grid */}
        {!loading && books.length > 0 && (
          <motion.div
            variants={staggerContainer}
            initial="hidden"
            animate="visible"
            className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6"
          >
            <AnimatePresence>
              {books.map((book, i) => (
                <motion.div
                  key={book.book_title}
                  layout
                  exit={{ opacity: 0, scale: 0.8, transition: { duration: 0.3 } }}
                >
                  <BookCard
                    title={book.book_title}
                    author={book.book_author}
                    image={book.book_image}
                    index={i}
                    isWishlisted={true}
                    onWishlistToggle={() => removeBook(book.book_title)}
                    onClick={() => navigate(`/book/${encodeURIComponent(book.book_title)}`)}
                  />
                </motion.div>
              ))}
            </AnimatePresence>
          </motion.div>
        )}
      </div>
    </div>
  );
}

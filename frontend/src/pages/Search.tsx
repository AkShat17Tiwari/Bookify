import { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { motion } from 'framer-motion';
import { staggerContainer, fadeUp } from '../lib/animations';
import SearchBar from '../components/UI/SearchBar';
import BookCard from '../components/UI/BookCard';
import GenreCard from '../components/UI/GenreCard';
import SkeletonLoader from '../components/UI/SkeletonLoader';
import { useApi } from '../lib/api';
import type { Book } from '../lib/api';
import { RiSparklingFill } from 'react-icons/ri';
import { HiOutlineEmojiSad, HiOutlineSearch } from 'react-icons/hi';

const MOOD_EMOJIS = [
  { emotion: 'happy', emoji: '😊', label: 'Happy' },
  { emotion: 'sad', emoji: '😢', label: 'Sad' },
  { emotion: 'neutral', emoji: '😐', label: 'Calm' },
  { emotion: 'surprised', emoji: '😲', label: 'Curious' },
  { emotion: 'fearful', emoji: '😰', label: 'Anxious' },
  { emotion: 'angry', emoji: '😤', label: 'Intense' },
];

const ALL_GENRES = [
  'Science Fiction', 'Romance', 'Mystery/Thriller', 'Fantasy', 'Horror',
  'Literary Fiction', 'Non-Fiction', 'Biography', 'Self-Help', 'History',
  'Classics', 'Poetry', 'Young Adult', 'Children', 'Cooking', 'Travel',
  'Fiction', 'Religious/Spiritual',
];

export default function Search() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const api = useApi();
  const [results, setResults] = useState<Book[]>([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const [matchInfo, setMatchInfo] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    const genre = searchParams.get('genre');
    if (genre) {
      handleSearch(`📂 Genre: ${genre}`, 'classic');
    }
  }, [searchParams]);

  const handleSearch = async (query: string, mode: string) => {
    setLoading(true);
    setSearched(true);
    setError('');
    setMatchInfo('');

    try {
      const data = await api.recommend(query, mode);
      setResults(data.data || []);
      if (data.matched_title) setMatchInfo(`Showing recommendations similar to "${data.matched_title}"`);
      if (data.genre_mode && data.matched_genre) setMatchInfo(`${data.matched_genre}`);
      if (!data.data || data.data.length === 0) setError('No books found. Try a different search term or genre.');
    } catch (err: any) {
      console.error('Search error:', err);
      if (err.message === 'AUTH_REQUIRED') {
        setError('Please sign in to search. Redirecting...');
        setTimeout(() => navigate('/auth'), 1500);
      } else {
        setError('Could not connect to the recommendation engine. Make sure the backend server is running.');
      }
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const handleMoodSearch = async (emotion: string) => {
    setLoading(true);
    setSearched(true);
    setError('');
    setMatchInfo(`Books for your ${emotion} mood`);

    try {
      const data = await api.moodRecommend(emotion);
      const books = (data.books || []).map((b: any) => {
        // Handle both array format and object format
        if (Array.isArray(b)) {
          return { title: b[0], author: b[1], image: b[2], reasons: b[3] };
        }
        return b;
      });
      setResults(books);
      if (books.length === 0) setError('No mood-based recommendations found.');
    } catch (err: any) {
      if (err.message === 'AUTH_REQUIRED') {
        setError('Please sign in for mood-based recommendations.');
      } else {
        setError('Could not fetch mood recommendations. Make sure the backend is running.');
      }
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen px-6 py-12">
      <div className="max-w-[1400px] mx-auto">
        {/* Header */}
        <motion.div
          variants={staggerContainer}
          initial="hidden"
          animate="visible"
          className="text-center mb-10"
        >
          <motion.div variants={fadeUp} custom={0} className="mb-4">
            <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-xs font-bold uppercase tracking-widest"
              style={{
                background: 'rgba(139,92,246,0.06)',
                border: '1px solid rgba(139,92,246,0.12)',
                color: '#8B5CF6',
              }}
            >
              <RiSparklingFill /> AI-Powered Discovery
            </span>
          </motion.div>
          <motion.h1 variants={fadeUp} custom={1} className="text-4xl md:text-5xl font-extrabold tracking-tight mb-4">
            Find Your Perfect <span className="gradient-text">Read</span>
          </motion.h1>
          <motion.p variants={fadeUp} custom={2} className="text-text-secondary text-lg max-w-lg mx-auto">
            Search by title, browse genres, or let AI match your mood
          </motion.p>
        </motion.div>

        {/* Search Bar */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="mb-12"
        >
          <SearchBar onSearch={handleSearch} />
        </motion.div>

        {/* Mood Search (when no results shown) */}
        {!searched && (
          <>
            <motion.div
              variants={staggerContainer}
              initial="hidden"
              animate="visible"
              className="mb-16"
            >
              <motion.h3 variants={fadeUp} custom={0} className="text-lg font-bold text-text-primary mb-5 text-center">
                😊 Search by Mood
              </motion.h3>
              <motion.div variants={fadeUp} custom={1} className="flex flex-wrap justify-center gap-3">
                {MOOD_EMOJIS.map((mood) => (
                  <motion.button
                    key={mood.emotion}
                    whileHover={{ y: -4, scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => handleMoodSearch(mood.emotion)}
                    className="px-5 py-3 rounded-[16px] flex items-center gap-2.5 text-sm font-semibold transition-all"
                    style={{
                      background: 'linear-gradient(145deg, #ffffff, #f5f3ee)',
                      boxShadow: '0 4px 12px rgba(0,0,0,0.05), inset 0 1px 0 rgba(255,255,255,0.9)',
                      border: '1px solid rgba(0,0,0,0.04)',
                    }}
                  >
                    <span className="text-xl">{mood.emoji}</span>
                    {mood.label}
                  </motion.button>
                ))}
              </motion.div>
            </motion.div>

            {/* Genre Grid */}
            <motion.div
              variants={staggerContainer}
              initial="hidden"
              animate="visible"
            >
              <motion.h3 variants={fadeUp} custom={0} className="text-lg font-bold text-text-primary mb-5 text-center">
                📂 Browse by Genre
              </motion.h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3 max-w-4xl mx-auto">
                {ALL_GENRES.map((genre, i) => (
                  <GenreCard
                    key={genre}
                    genre={genre}
                    index={i}
                    onClick={() => handleSearch(`📂 Genre: ${genre}`, 'classic')}
                  />
                ))}
              </div>
            </motion.div>
          </>
        )}

        {/* Loading */}
        {loading && (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-6">
            <SkeletonLoader variant="card" count={8} />
          </div>
        )}

        {/* Results */}
        {!loading && searched && (
          <div>
            {matchInfo && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-6 flex items-center gap-2"
              >
                <HiOutlineSearch className="text-bookify-blue" />
                <span className="text-sm font-semibold text-text-secondary">{matchInfo}</span>
                <span className="text-xs text-text-muted ml-1">({results.length} results)</span>
              </motion.div>
            )}

            {error ? (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-center py-20"
              >
                <HiOutlineEmojiSad className="text-5xl text-text-muted mx-auto mb-4" />
                <p className="text-text-secondary font-medium">{error}</p>
              </motion.div>
            ) : (
              <motion.div
                variants={staggerContainer}
                initial="hidden"
                animate="visible"
                className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-6"
              >
                {results.map((book, i) => (
                  <BookCard
                    key={book.title}
                    title={book.title}
                    author={book.author}
                    image={book.image}
                    reasons={book.reasons}
                    index={i}
                    onClick={() => navigate(`/book/${encodeURIComponent(book.title)}`)}
                  />
                ))}
              </motion.div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

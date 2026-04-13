
import { useState, useEffect, useRef } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { staggerContainer, fadeUp } from '../lib/animations';
import { useParallax, useCountUp, useScrollReveal } from '../hooks/useScrollReveal';
import BookCard from '../components/UI/BookCard';
import GenreCard from '../components/UI/GenreCard';
import SkeletonLoader from '../components/UI/SkeletonLoader';
import Button from '../components/UI/Button';
import { useApi } from '../lib/api';
import type { PopularBook } from '../lib/api';

import { HiOutlineSparkles, HiOutlineLightningBolt, HiOutlineChartBar } from 'react-icons/hi';
import { RiSparklingFill, RiBrainLine, RiBookOpenLine } from 'react-icons/ri';

// Fallback data in case the backend is not running
const FALLBACK_BOOKS: PopularBook[] = [
  { title: 'The Lovely Bones', author: 'Alice Sebold', image: 'https://covers.openlibrary.org/b/isbn/0316666343-M.jpg', rating: 4.2, votes: 1260 },
  { title: 'Wild Animus', author: 'Rich Shapero', image: 'https://covers.openlibrary.org/b/isbn/0971880107-M.jpg', rating: 3.8, votes: 980 },
  { title: 'The Da Vinci Code', author: 'Dan Brown', image: 'https://covers.openlibrary.org/b/isbn/0385504209-M.jpg', rating: 4.5, votes: 2100 },
  { title: 'A Painted House', author: 'John Grisham', image: 'https://covers.openlibrary.org/b/isbn/0385337817-M.jpg', rating: 4.1, votes: 870 },
  { title: 'The Secret Life of Bees', author: 'Sue Monk Kidd', image: 'https://covers.openlibrary.org/b/isbn/0142001740-M.jpg', rating: 4.3, votes: 1150 },
  { title: 'Divine Secrets of the Ya-Ya Sisterhood', author: 'Rebecca Wells', image: 'https://covers.openlibrary.org/b/isbn/0060928328-M.jpg', rating: 4.0, votes: 760 },
  { title: 'The Red Tent', author: 'Anita Diamant', image: 'https://covers.openlibrary.org/b/isbn/0312195516-M.jpg', rating: 4.4, votes: 990 },
  { title: 'Where the Heart Is', author: 'Billie Letts', image: 'https://covers.openlibrary.org/b/isbn/0446672211-M.jpg', rating: 4.2, votes: 830 },
];

const GENRES = [
  'Science Fiction', 'Romance', 'Mystery/Thriller', 'Fantasy',
  'Literary Fiction', 'Biography', 'Self-Help', 'History',
];

export default function Landing() {
  const parallaxOffset = useParallax(0.15);
  const statsRef = useScrollReveal(0.3);
  const navigate = useNavigate();
  const api = useApi();
  const apiRef = useRef(api);
  apiRef.current = api;

  // Live data state
  const [popularBooks, setPopularBooks] = useState<PopularBook[]>([]);
  const [stats, setStats] = useState({ total_books: 4893, total_genres: 23, total_users: 1247 });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchPopular = async () => {
      try {
        const data = await apiRef.current.getPopularBooks();
        if (data.books && data.books.length > 0) {
          setPopularBooks(data.books.slice(0, 8));
        } else {
          setPopularBooks(FALLBACK_BOOKS);
        }
        if (data.stats) {
          setStats(data.stats);
        }
      } catch (err) {
        console.warn('Backend not available, using fallback data:', err);
        setPopularBooks(FALLBACK_BOOKS);
      } finally {
        setLoading(false);
      }
    };
    fetchPopular();
  }, []);

  const bookCount = useCountUp(stats.total_books, 2000, statsRef.isVisible);
  const userCount = useCountUp(stats.total_users, 2000, statsRef.isVisible);
  const genreCount = useCountUp(stats.total_genres, 1500, statsRef.isVisible);

  return (
    <div className="relative overflow-hidden">
      {/* ═══ HERO SECTION ═══ */}
      <section className="relative min-h-[90vh] flex items-center justify-center px-6 overflow-hidden">
        {/* Background decorative elements */}
        <div className="absolute inset-0 pointer-events-none overflow-hidden">
          {/* Radial gradient */}
          <div className="absolute top-[-200px] left-1/2 -translate-x-1/2 w-[900px] h-[600px] opacity-40"
            style={{ background: 'radial-gradient(ellipse, rgba(74,144,217,0.12) 0%, rgba(139,92,246,0.08) 40%, transparent 70%)' }}
          />

          {/* Floating books */}
          <motion.div
            animate={{ y: [0, -20, 0], rotate: [0, 3, 0] }}
            transition={{ duration: 6, repeat: Infinity, ease: 'easeInOut' }}
            className="absolute top-[15%] left-[8%] w-20 h-28 rounded-[12px] opacity-20"
            style={{
              background: 'linear-gradient(135deg, #4A90D9, #6366F1)',
              boxShadow: '0 10px 30px rgba(74,144,217,0.2)',
              transform: `translateY(${parallaxOffset}px) perspective(500px) rotateY(-15deg)`,
            }}
          />
          <motion.div
            animate={{ y: [0, -15, 0], rotate: [0, -2, 0] }}
            transition={{ duration: 8, repeat: Infinity, ease: 'easeInOut', delay: 1 }}
            className="absolute top-[20%] right-[10%] w-16 h-24 rounded-[10px] opacity-15"
            style={{
              background: 'linear-gradient(135deg, #F43F5E, #F59E0B)',
              boxShadow: '0 10px 30px rgba(244,63,94,0.15)',
              transform: `translateY(${parallaxOffset * 0.7}px) perspective(500px) rotateY(15deg)`,
            }}
          />
          <motion.div
            animate={{ y: [0, -18, 0], rotate: [0, 1.5, 0] }}
            transition={{ duration: 7, repeat: Infinity, ease: 'easeInOut', delay: 2 }}
            className="absolute bottom-[25%] left-[15%] w-14 h-20 rounded-[8px] opacity-10"
            style={{
              background: 'linear-gradient(135deg, #14B8A6, #4A90D9)',
              boxShadow: '0 8px 25px rgba(20,184,166,0.15)',
              transform: `translateY(${parallaxOffset * 1.2}px) perspective(500px) rotateY(-10deg)`,
            }}
          />
          <motion.div
            animate={{ y: [0, -12, 0] }}
            transition={{ duration: 9, repeat: Infinity, ease: 'easeInOut', delay: 0.5 }}
            className="absolute bottom-[30%] right-[12%] w-18 h-26 rounded-[10px] opacity-12"
            style={{
              background: 'linear-gradient(135deg, #8B5CF6, #EC4899)',
              transform: `translateY(${parallaxOffset * 0.9}px) perspective(500px) rotateY(12deg)`,
            }}
          />

          {/* Sparkle particles */}
          {Array.from({ length: 8 }, (_, i) => (
            <motion.div
              key={i}
              className="absolute w-1.5 h-1.5 rounded-full"
              style={{
                background: i % 2 === 0 ? '#4A90D9' : '#8B5CF6',
                left: `${15 + i * 10}%`,
                top: `${20 + (i % 3) * 25}%`,
              }}
              animate={{
                opacity: [0, 1, 0],
                scale: [0, 1, 0],
                rotate: [0, 180, 360],
              }}
              transition={{
                duration: 3,
                repeat: Infinity,
                delay: i * 0.4,
                ease: 'easeInOut',
              }}
            />
          ))}
        </div>

        {/* Hero Content */}
        <motion.div
          variants={staggerContainer}
          initial="hidden"
          animate="visible"
          className="relative z-10 text-center max-w-3xl"
        >
          {/* Badge */}
          <motion.div variants={fadeUp} custom={0} className="mb-6">
            <span className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full text-xs font-bold uppercase tracking-widest"
              style={{
                background: 'linear-gradient(135deg, rgba(74,144,217,0.08), rgba(139,92,246,0.06))',
                border: '1px solid rgba(74,144,217,0.15)',
                color: '#4A90D9',
              }}
            >
              <RiSparklingFill className="text-bookify-purple animate-pulse" />
              AI-Powered Book Discovery
            </span>
          </motion.div>

          {/* Title */}
          <motion.h1
            variants={fadeUp}
            custom={1}
            className="text-5xl md:text-7xl font-black tracking-[-0.03em] leading-[1.05] mb-6"
          >
            Discover Your Next
            <br />
            <span className="gradient-text">Favorite Book</span>
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            variants={fadeUp}
            custom={2}
            className="text-lg md:text-xl text-text-secondary max-w-xl mx-auto mb-10 leading-relaxed"
          >
            Intelligent recommendations from thousands of reader reviews.
            Find books you'll love with our AI-powered discovery engine.
          </motion.p>

          {/* CTA Buttons */}
          <motion.div variants={fadeUp} custom={3} className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Button
              variant="glow"
              size="lg"
              onClick={() => navigate('/search')}
              className="animate-glow-pulse min-w-[200px]"
            >
              <HiOutlineSparkles className="text-lg" />
              Start Discovering
            </Button>
            <Button
              variant="secondary"
              size="lg"
              onClick={() => navigate('/dashboard')}
              className="min-w-[200px]"
            >
              <HiOutlineChartBar />
              View Dashboard
            </Button>
          </motion.div>

          {/* Quick Stats */}
          <motion.div
            variants={fadeUp}
            custom={4}
            ref={statsRef.ref}
            className="mt-12 flex items-center justify-center gap-8 md:gap-12"
          >
            {[
              { value: bookCount.toLocaleString(), label: 'Books', icon: '📚' },
              { value: userCount.toLocaleString(), label: 'Users', icon: '👥' },
              { value: genreCount, label: 'Genres', icon: '📂' },
            ].map((stat, i) => (
              <div key={i} className="text-center">
                <div className="text-2xl md:text-3xl font-extrabold text-text-primary">
                  {stat.icon} {stat.value}
                </div>
                <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mt-1">
                  {stat.label}
                </div>
              </div>
            ))}
          </motion.div>
        </motion.div>
      </section>

      {/* ═══ HOW IT WORKS ═══ */}
      <section className="py-24 px-6">
        <div className="max-w-[1200px] mx-auto">
          <motion.div
            variants={staggerContainer}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: '-100px' }}
            className="text-center mb-16"
          >
            <motion.span variants={fadeUp} custom={0} className="text-xs font-bold uppercase tracking-widest text-bookify-purple mb-3 block">
              How It Works
            </motion.span>
            <motion.h2 variants={fadeUp} custom={1} className="text-3xl md:text-4xl font-extrabold tracking-tight">
              Smart Recommendations in <span className="gradient-text">3 Steps</span>
            </motion.h2>
          </motion.div>

          <motion.div
            variants={staggerContainer}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="grid grid-cols-1 md:grid-cols-3 gap-8"
          >
            {[
              {
                icon: RiBookOpenLine,
                title: 'Tell Us What You Like',
                desc: 'Search for a book, genre, or describe your mood. Our AI understands natural language.',
                color: '#4A90D9',
                bgColor: 'rgba(74,144,217,0.06)',
              },
              {
                icon: RiBrainLine,
                title: 'AI Analyzes Patterns',
                desc: 'Neural networks analyze reading patterns from thousands of users to find perfect matches.',
                color: '#8B5CF6',
                bgColor: 'rgba(139,92,246,0.06)',
              },
              {
                icon: HiOutlineLightningBolt,
                title: 'Get Personalized Results',
                desc: 'Receive curated recommendations with explainable AI insights for each pick.',
                color: '#14B8A6',
                bgColor: 'rgba(20,184,166,0.06)',
              },
            ].map((step, i) => (
              <motion.div
                key={i}
                variants={fadeUp}
                custom={i}
                whileHover={{ y: -8 }}
                className="relative rounded-[24px] p-8 text-center"
                style={{
                  background: 'linear-gradient(145deg, #ffffff, #f8f6f2)',
                  boxShadow: '0 8px 30px rgba(0,0,0,0.05), 0 0 0 1px rgba(0,0,0,0.02), inset 0 2px 0 rgba(255,255,255,1)',
                }}
              >
                {/* Step number */}
                <div className="absolute -top-3 -right-3 w-8 h-8 rounded-full flex items-center justify-center text-xs font-extrabold text-white"
                  style={{ background: `linear-gradient(135deg, ${step.color}, ${step.color}dd)`, boxShadow: `0 4px 12px ${step.color}33` }}
                >
                  {i + 1}
                </div>

                <div className="w-16 h-16 rounded-[18px] flex items-center justify-center mx-auto mb-5"
                  style={{ background: step.bgColor }}
                >
                  <step.icon className="text-2xl" style={{ color: step.color }} />
                </div>
                <h3 className="text-lg font-bold text-text-primary mb-3">{step.title}</h3>
                <p className="text-sm text-text-tertiary leading-relaxed">{step.desc}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* ═══ TRENDING BOOKS ═══ */}
      <section className="py-24 px-6">
        <div className="max-w-[1400px] mx-auto">
          <motion.div
            variants={staggerContainer}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: '-100px' }}
            className="flex items-end justify-between mb-12"
          >
            <div>
              <motion.span variants={fadeUp} custom={0} className="text-xs font-bold uppercase tracking-widest text-bookify-amber mb-2 block">
                🔥 Trending Now
              </motion.span>
              <motion.h2 variants={fadeUp} custom={1} className="text-3xl md:text-4xl font-extrabold tracking-tight">
                Popular with Readers
              </motion.h2>
            </div>
            <motion.div variants={fadeUp} custom={2}>
              <Link to="/search">
                <Button variant="ghost" size="sm">View all →</Button>
              </Link>
            </motion.div>
          </motion.div>

          {loading ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-6">
              <SkeletonLoader variant="card" count={8} />
            </div>
          ) : (
            <motion.div
              variants={staggerContainer}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-6"
            >
              {popularBooks.map((book, i) => (
                <BookCard
                  key={book.title}
                  title={book.title}
                  author={book.author}
                  image={book.image}
                  rating={book.rating}
                  votes={book.votes}
                  index={i}
                  onClick={() => navigate(`/book/${encodeURIComponent(book.title)}`)}
                />
              ))}
            </motion.div>
          )}
        </div>
      </section>

      {/* ═══ GENRE EXPLORER ═══ */}
      <section className="py-24 px-6" style={{
        background: 'linear-gradient(180deg, transparent, rgba(74,144,217,0.02), transparent)',
      }}>
        <div className="max-w-[1200px] mx-auto">
          <motion.div
            variants={staggerContainer}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: '-100px' }}
            className="text-center mb-14"
          >
            <motion.span variants={fadeUp} custom={0} className="text-xs font-bold uppercase tracking-widest text-bookify-teal mb-3 block">
              Explore Genres
            </motion.span>
            <motion.h2 variants={fadeUp} custom={1} className="text-3xl md:text-4xl font-extrabold tracking-tight">
              Browse by <span className="gradient-text">Genre</span>
            </motion.h2>
          </motion.div>

          <motion.div
            variants={staggerContainer}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4"
          >
            {GENRES.map((genre, i) => (
              <GenreCard
                key={genre}
                genre={genre}
                index={i}
                onClick={() => navigate(`/search?genre=${encodeURIComponent(genre)}`)}
              />
            ))}
          </motion.div>
        </div>
      </section>

      {/* ═══ CTA SECTION ═══ */}
      <section className="py-32 px-6">
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="max-w-3xl mx-auto text-center rounded-[32px] p-16 relative overflow-hidden"
          style={{
            background: 'linear-gradient(135deg, #4A90D9, #8B5CF6)',
            boxShadow: '0 20px 60px rgba(74,144,217,0.25), inset 0 2px 0 rgba(255,255,255,0.15)',
          }}
        >
          {/* Background pattern */}
          <div className="absolute inset-0 opacity-10">
            {Array.from({ length: 12 }, (_, i) => (
              <div
                key={i}
                className="absolute rounded-full"
                style={{
                  width: 40 + Math.random() * 80,
                  height: 40 + Math.random() * 80,
                  left: `${Math.random() * 100}%`,
                  top: `${Math.random() * 100}%`,
                  background: 'rgba(255,255,255,0.1)',
                  transform: `rotate(${Math.random() * 360}deg)`,
                }}
              />
            ))}
          </div>

          <div className="relative z-10">
            <h2 className="text-3xl md:text-4xl font-extrabold text-white mb-4 tracking-tight">
              Ready to find your next read?
            </h2>
            <p className="text-lg text-white/80 mb-8 max-w-md mx-auto">
              Join thousands of readers discovering books they love through AI-powered recommendations.
            </p>
            <Button
              variant="secondary"
              size="lg"
              onClick={() => navigate('/search')}
              className="!bg-white !text-bookify-blue font-extrabold"
            >
              <HiOutlineSparkles />
              Explore Now — It's Free
            </Button>
          </div>
        </motion.div>
      </section>
    </div>
  );
}

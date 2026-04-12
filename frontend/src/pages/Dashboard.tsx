import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { staggerContainer, fadeUp } from '../lib/animations';
import BookCard from '../components/UI/BookCard';
import ProgressRing from '../components/UI/ProgressRing';
import GenreCard from '../components/UI/GenreCard';

import Button from '../components/UI/Button';
import { HiOutlineBookOpen, HiOutlineHeart, HiOutlineClock, HiOutlineSearch, HiOutlineChartBar, HiOutlineFire } from 'react-icons/hi';
import { RiSparklingFill } from 'react-icons/ri';

const SAMPLE_FOR_YOU = [
  { title: 'The Kite Runner', author: 'Khaled Hosseini', image: 'https://covers.openlibrary.org/b/isbn/1594631931-M.jpg', rating: 4.6 },
  { title: 'Life of Pi', author: 'Yann Martel', image: 'https://covers.openlibrary.org/b/isbn/0156027321-M.jpg', rating: 4.3 },
  { title: 'The Alchemist', author: 'Paulo Coelho', image: 'https://covers.openlibrary.org/b/isbn/0061122416-M.jpg', rating: 4.5 },
  { title: 'Memoirs of a Geisha', author: 'Arthur Golden', image: 'https://covers.openlibrary.org/b/isbn/0375700439-M.jpg', rating: 4.2 },
];

const RECENT_GENRES = ['Literary Fiction', 'Mystery/Thriller', 'Romance', 'Science Fiction'];

export default function Dashboard() {
  const navigate = useNavigate();


  const stats = [
    { label: 'Books Read', value: '24', icon: HiOutlineBookOpen, color: '#4A90D9', bg: 'rgba(74,144,217,0.06)' },
    { label: 'Wishlist', value: '12', icon: HiOutlineHeart, color: '#F43F5E', bg: 'rgba(244,63,94,0.06)' },
    { label: 'This Month', value: '5', icon: HiOutlineClock, color: '#8B5CF6', bg: 'rgba(139,92,246,0.06)' },
    { label: 'Avg Rating', value: '4.3', icon: HiOutlineFire, color: '#F59E0B', bg: 'rgba(245,158,11,0.06)' },
  ];

  return (
    <div className="min-h-screen px-6 py-10">
      <div className="max-w-[1400px] mx-auto">
        {/* Header */}
        <motion.div
          variants={staggerContainer}
          initial="hidden"
          animate="visible"
          className="mb-10"
        >
          <motion.div variants={fadeUp} custom={0} className="flex items-center gap-3 mb-2">
            <div className="w-12 h-12 rounded-[16px] flex items-center justify-center text-2xl"
              style={{
                background: 'linear-gradient(135deg, #4A90D9, #8B5CF6)',
                boxShadow: '0 6px 20px rgba(74,144,217,0.25)',
              }}
            >
              👋
            </div>
            <div>
              <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight">Welcome back!</h1>
              <p className="text-sm text-text-tertiary">Here's your reading journey at a glance</p>
            </div>
          </motion.div>
        </motion.div>

        {/* Stats Cards */}
        <motion.div
          variants={staggerContainer}
          initial="hidden"
          animate="visible"
          className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10"
        >
          {stats.map((stat, i) => (
            <motion.div
              key={stat.label}
              variants={fadeUp}
              custom={i}
              whileHover={{ y: -4 }}
              className="rounded-[20px] p-5"
              style={{
                background: 'linear-gradient(145deg, #ffffff, #f8f6f2)',
                boxShadow: '0 4px 15px rgba(0,0,0,0.04), 0 0 0 1px rgba(0,0,0,0.02), inset 0 1px 0 rgba(255,255,255,0.9)',
              }}
            >
              <div className="w-10 h-10 rounded-[12px] flex items-center justify-center mb-3"
                style={{ background: stat.bg }}
              >
                <stat.icon className="text-lg" style={{ color: stat.color }} />
              </div>
              <div className="text-2xl font-extrabold text-text-primary">{stat.value}</div>
              <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mt-0.5">{stat.label}</div>
            </motion.div>
          ))}
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-10">
            {/* For You Recommendations */}
            <motion.div
              variants={staggerContainer}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
            >
              <div className="flex items-center justify-between mb-6">
                <motion.div variants={fadeUp} custom={0} className="flex items-center gap-2">
                  <RiSparklingFill className="text-bookify-purple" />
                  <h2 className="text-xl font-bold tracking-tight">Recommended for You</h2>
                </motion.div>
                <Link to="/search">
                  <Button variant="ghost" size="sm">See all →</Button>
                </Link>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {SAMPLE_FOR_YOU.map((book, i) => (
                  <BookCard
                    key={book.title}
                    title={book.title}
                    author={book.author}
                    image={book.image}
                    rating={book.rating}
                    index={i}
                    onClick={() => navigate(`/book/${encodeURIComponent(book.title)}`)}
                  />
                ))}
              </div>
            </motion.div>

            {/* Quick Actions */}
            <motion.div
              variants={staggerContainer}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
            >
              <motion.h2 variants={fadeUp} className="text-xl font-bold tracking-tight mb-5">Quick Actions</motion.h2>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                {[
                  { label: 'Discover Books', desc: 'Search & explore', icon: HiOutlineSearch, path: '/search', color: '#4A90D9', bg: 'rgba(74,144,217,0.06)' },
                  { label: 'My Wishlist', desc: 'Saved for later', icon: HiOutlineHeart, path: '/wishlist', color: '#F43F5E', bg: 'rgba(244,63,94,0.06)' },
                  { label: 'Analytics', desc: 'Reading insights', icon: HiOutlineChartBar, path: '/history', color: '#8B5CF6', bg: 'rgba(139,92,246,0.06)' },
                ].map((action, i) => (
                  <motion.div
                    key={action.label}
                    variants={fadeUp}
                    custom={i}
                    whileHover={{ y: -4 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => navigate(action.path)}
                    className="cursor-pointer rounded-[20px] p-6 flex items-center gap-4"
                    style={{
                      background: 'linear-gradient(145deg, #ffffff, #f8f6f2)',
                      boxShadow: '0 4px 15px rgba(0,0,0,0.04), inset 0 1px 0 rgba(255,255,255,0.9)',
                      border: '1px solid rgba(0,0,0,0.03)',
                    }}
                  >
                    <div className="w-12 h-12 rounded-[14px] flex items-center justify-center flex-shrink-0"
                      style={{ background: action.bg }}
                    >
                      <action.icon className="text-xl" style={{ color: action.color }} />
                    </div>
                    <div>
                      <h3 className="text-sm font-bold text-text-primary">{action.label}</h3>
                      <p className="text-xs text-text-muted">{action.desc}</p>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Reading Goal */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="rounded-[22px] p-6 text-center"
              style={{
                background: 'linear-gradient(145deg, #ffffff, #f8f6f2)',
                boxShadow: '0 6px 20px rgba(0,0,0,0.05), inset 0 1px 0 rgba(255,255,255,0.9)',
              }}
            >
              <h3 className="text-sm font-bold text-text-primary mb-4">📚 Reading Goal</h3>
              <ProgressRing progress={68} size={130} label="Complete" sublabel="24 of 36 books" />
              <p className="text-xs text-text-muted mt-3">12 books to go this year!</p>
            </motion.div>

            {/* Top Genres */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="rounded-[22px] p-6"
              style={{
                background: 'linear-gradient(145deg, #ffffff, #f8f6f2)',
                boxShadow: '0 6px 20px rgba(0,0,0,0.05), inset 0 1px 0 rgba(255,255,255,0.9)',
              }}
            >
              <h3 className="text-sm font-bold text-text-primary mb-4">🎯 Your Top Genres</h3>
              <div className="grid grid-cols-2 gap-2.5">
                {RECENT_GENRES.map((genre, i) => (
                  <GenreCard key={genre} genre={genre} index={i} onClick={() => navigate(`/search?genre=${encodeURIComponent(genre)}`)} />
                ))}
              </div>
            </motion.div>

            {/* Reading Streak */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
              className="rounded-[22px] p-6"
              style={{
                background: 'linear-gradient(135deg, #4A90D9, #8B5CF6)',
                boxShadow: '0 10px 30px rgba(74,144,217,0.2)',
              }}
            >
              <div className="text-white">
                <div className="text-4xl mb-2">🔥</div>
                <div className="text-3xl font-extrabold">7 Days</div>
                <div className="text-sm text-white/70 font-medium mt-1">Reading Streak</div>
                <div className="flex gap-1.5 mt-3">
                  {Array.from({ length: 7 }, (_, i) => (
                    <div key={i} className="w-6 h-6 rounded-full bg-white/20 flex items-center justify-center text-xs">
                      ✓
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
}

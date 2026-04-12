
import { motion } from 'framer-motion';
import { staggerContainer, fadeUp } from '../lib/animations';
import ProgressRing from '../components/UI/ProgressRing';

import { PieChart, Pie, Cell, XAxis, YAxis, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { HiOutlineChartBar, HiOutlineBookOpen, HiOutlineCalendar, HiOutlineStar, HiOutlineTrendingUp } from 'react-icons/hi';

const GENRE_COLORS = ['#4A90D9', '#8B5CF6', '#14B8A6', '#F59E0B', '#F43F5E', '#6366F1', '#EC4899', '#10B981'];

const SAMPLE_GENRE_DATA = [
  { name: 'Literary Fiction', value: 8 },
  { name: 'Mystery/Thriller', value: 6 },
  { name: 'Romance', value: 5 },
  { name: 'Science Fiction', value: 4 },
  { name: 'Biography', value: 3 },
  { name: 'Self-Help', value: 2 },
];

const SAMPLE_MONTHLY_DATA = [
  { month: 'Oct', books: 2 },
  { month: 'Nov', books: 3 },
  { month: 'Dec', books: 5 },
  { month: 'Jan', books: 4 },
  { month: 'Feb', books: 3 },
  { month: 'Mar', books: 6 },
  { month: 'Apr', books: 5 },
];

const SAMPLE_ACTIVITY = Array.from({ length: 52 * 7 }, (_, i) => ({
  day: i,
  value: Math.random() > 0.5 ? Math.floor(Math.random() * 4) : 0,
}));

export default function ReadingHistory() {


  const totalBooks = 28;
  const totalGenres = 6;
  const avgRating = 4.2;
  const thisYear = 24;

  const styledTooltip = {
    contentStyle: {
      background: 'rgba(255,255,255,0.95)',
      backdropFilter: 'blur(20px)',
      border: '1px solid rgba(0,0,0,0.05)',
      borderRadius: '14px',
      boxShadow: '0 8px 30px rgba(0,0,0,0.08)',
      padding: '10px 14px',
      fontSize: '13px',
      fontWeight: 600,
    },
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
              style={{ background: 'linear-gradient(135deg, rgba(139,92,246,0.1), rgba(139,92,246,0.05))', border: '1px solid rgba(139,92,246,0.15)' }}
            >
              <HiOutlineChartBar className="text-xl text-bookify-purple" />
            </div>
            <div>
              <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight">Reading Analytics</h1>
              <p className="text-sm text-text-tertiary">Your reading journey visualized</p>
            </div>
          </motion.div>
        </motion.div>

        {/* Stats Row */}
        <motion.div
          variants={staggerContainer}
          initial="hidden"
          animate="visible"
          className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10"
        >
          {[
            { label: 'Total Books', value: totalBooks, icon: HiOutlineBookOpen, color: '#4A90D9', bg: 'rgba(74,144,217,0.06)' },
            { label: 'Genres Read', value: totalGenres, icon: HiOutlineCalendar, color: '#8B5CF6', bg: 'rgba(139,92,246,0.06)' },
            { label: 'Avg Rating', value: avgRating, icon: HiOutlineStar, color: '#F59E0B', bg: 'rgba(245,158,11,0.06)' },
            { label: 'This Year', value: thisYear, icon: HiOutlineTrendingUp, color: '#14B8A6', bg: 'rgba(20,184,166,0.06)' },
          ].map((stat, i) => (
            <motion.div
              key={stat.label}
              variants={fadeUp}
              custom={i}
              whileHover={{ y: -4 }}
              className="rounded-[20px] p-5"
              style={{
                background: 'linear-gradient(145deg, #ffffff, #f8f6f2)',
                boxShadow: '0 4px 15px rgba(0,0,0,0.04), inset 0 1px 0 rgba(255,255,255,0.9)',
              }}
            >
              <div className="w-10 h-10 rounded-[12px] flex items-center justify-center mb-3" style={{ background: stat.bg }}>
                <stat.icon className="text-lg" style={{ color: stat.color }} />
              </div>
              <div className="text-2xl font-extrabold text-text-primary">{stat.value}</div>
              <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mt-0.5">{stat.label}</div>
            </motion.div>
          ))}
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Genre Distribution */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="rounded-[24px] p-6"
            style={{
              background: 'linear-gradient(145deg, #ffffff, #f8f6f2)',
              boxShadow: '0 8px 30px rgba(0,0,0,0.05), inset 0 1px 0 rgba(255,255,255,0.9)',
            }}
          >
            <h3 className="text-lg font-bold text-text-primary mb-6">📊 Genre Distribution</h3>
            <div className="flex items-center gap-8">
              <ResponsiveContainer width="50%" height={200}>
                <PieChart>
                  <Pie
                    data={SAMPLE_GENRE_DATA}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={80}
                    paddingAngle={4}
                    dataKey="value"
                  >
                    {SAMPLE_GENRE_DATA.map((_, i) => (
                      <Cell key={i} fill={GENRE_COLORS[i % GENRE_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip {...styledTooltip} />
                </PieChart>
              </ResponsiveContainer>
              <div className="flex-1 space-y-2.5">
                {SAMPLE_GENRE_DATA.map((entry, i) => (
                  <div key={entry.name} className="flex items-center gap-2.5">
                    <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ background: GENRE_COLORS[i] }} />
                    <span className="text-xs font-semibold text-text-secondary flex-1 truncate">{entry.name}</span>
                    <span className="text-xs font-bold text-text-primary">{entry.value}</span>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>

          {/* Monthly Reading */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="rounded-[24px] p-6"
            style={{
              background: 'linear-gradient(145deg, #ffffff, #f8f6f2)',
              boxShadow: '0 8px 30px rgba(0,0,0,0.05), inset 0 1px 0 rgba(255,255,255,0.9)',
            }}
          >
            <h3 className="text-lg font-bold text-text-primary mb-6">📈 Books Per Month</h3>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={SAMPLE_MONTHLY_DATA}>
                <defs>
                  <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#4A90D9" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#4A90D9" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="month" tick={{ fontSize: 12, fontWeight: 600, fill: '#8888A8' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 12, fontWeight: 600, fill: '#8888A8' }} axisLine={false} tickLine={false} />
                <Tooltip {...styledTooltip} />
                <Area type="monotone" dataKey="books" stroke="#4A90D9" strokeWidth={3} fill="url(#areaGrad)" />
              </AreaChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Activity Heatmap */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="rounded-[24px] p-6 lg:col-span-2"
            style={{
              background: 'linear-gradient(145deg, #ffffff, #f8f6f2)',
              boxShadow: '0 8px 30px rgba(0,0,0,0.05), inset 0 1px 0 rgba(255,255,255,0.9)',
            }}
          >
            <h3 className="text-lg font-bold text-text-primary mb-6">🗓️ Reading Activity</h3>
            <div className="flex flex-wrap gap-[3px]">
              {SAMPLE_ACTIVITY.map((day, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, scale: 0 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: i * 0.001, duration: 0.2 }}
                  className="w-3 h-3 rounded-[3px]"
                  style={{
                    background: day.value === 0
                      ? 'rgba(0,0,0,0.04)'
                      : day.value === 1
                      ? 'rgba(74,144,217,0.2)'
                      : day.value === 2
                      ? 'rgba(74,144,217,0.45)'
                      : 'rgba(74,144,217,0.8)',
                  }}
                  title={`${day.value} books`}
                />
              ))}
            </div>
            <div className="flex items-center gap-1.5 mt-3 justify-end">
              <span className="text-[10px] text-text-muted font-semibold">Less</span>
              {[0.04, 0.2, 0.45, 0.8].map((opacity, i) => (
                <div key={i} className="w-3 h-3 rounded-[3px]" style={{ background: i === 0 ? 'rgba(0,0,0,0.04)' : `rgba(74,144,217,${opacity})` }} />
              ))}
              <span className="text-[10px] text-text-muted font-semibold">More</span>
            </div>
          </motion.div>

          {/* Reading Goals */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="rounded-[24px] p-6 lg:col-span-2"
            style={{
              background: 'linear-gradient(145deg, #ffffff, #f8f6f2)',
              boxShadow: '0 8px 30px rgba(0,0,0,0.05), inset 0 1px 0 rgba(255,255,255,0.9)',
            }}
          >
            <h3 className="text-lg font-bold text-text-primary mb-6">🎯 Reading Goals</h3>
            <div className="flex flex-wrap items-center justify-center gap-10">
              <ProgressRing progress={68} size={140} label="Yearly" sublabel="24/36 books" gradientId="ring1" />
              <ProgressRing progress={83} size={140} label="Monthly" sublabel="5/6 books" gradientId="ring2" />
              <ProgressRing progress={45} size={140} label="Genres" sublabel="6/13 explored" gradientId="ring3" />
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}

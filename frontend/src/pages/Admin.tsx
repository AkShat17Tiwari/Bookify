
import { motion } from 'framer-motion';
import { staggerContainer, fadeUp } from '../lib/animations';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { HiOutlineUsers, HiOutlineBookOpen, HiOutlineShieldCheck, HiOutlineClock, HiOutlineSearch } from 'react-icons/hi';

const USER_DATA = [
  { month: 'Jan', users: 120 },
  { month: 'Feb', users: 180 },
  { month: 'Mar', users: 250 },
  { month: 'Apr', users: 340 },
  { month: 'May', users: 410 },
  { month: 'Jun', users: 520 },
];

const GENRE_POPULARITY = [
  { name: 'Fiction', value: 35 },
  { name: 'Mystery', value: 25 },
  { name: 'Sci-Fi', value: 20 },
  { name: 'Romance', value: 15 },
  { name: 'Other', value: 5 },
];

const RECENT_SEARCH = [
  { time: '2 min ago', query: 'The Great Gatsby', type: 'Title', user: 'john@email.com' },
  { time: '5 min ago', query: 'Science Fiction', type: 'Genre', user: 'sarah@email.com' },
  { time: '8 min ago', query: 'Paulo Coelho', type: 'Author', user: 'mike@email.com' },
  { time: '12 min ago', query: 'Happy mood', type: 'Mood', user: 'anna@email.com' },
  { time: '15 min ago', query: 'Romance', type: 'Genre', user: 'bob@email.com' },
];

const AUDIT_EVENTS = [
  { time: '1 min ago', event: 'User login', level: 'info', user: 'john@email.com' },
  { time: '3 min ago', event: 'Book rated', level: 'info', user: 'sarah@email.com' },
  { time: '7 min ago', event: 'Wishlist updated', level: 'info', user: 'mike@email.com' },
  { time: '15 min ago', event: 'Failed login attempt', level: 'warning', user: 'unknown' },
  { time: '22 min ago', event: 'New user registered', level: 'success', user: 'newuser@email.com' },
  { time: '30 min ago', event: 'Admin access', level: 'info', user: 'admin@bookify.ai' },
];

const COLORS = ['#4A90D9', '#8B5CF6', '#14B8A6', '#F59E0B', '#F43F5E'];

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

export default function Admin() {
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
              style={{ background: 'linear-gradient(135deg, rgba(244,63,94,0.1), rgba(244,63,94,0.05))', border: '1px solid rgba(244,63,94,0.15)' }}
            >
              <HiOutlineShieldCheck className="text-xl text-bookify-rose" />
            </div>
            <div>
              <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight">Admin Panel</h1>
              <p className="text-sm text-text-tertiary">System analytics and management</p>
            </div>
          </motion.div>
        </motion.div>

        {/* Stats */}
        <motion.div
          variants={staggerContainer}
          initial="hidden"
          animate="visible"
          className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10"
        >
          {[
            { label: 'Total Users', value: '1,247', change: '+12%', icon: HiOutlineUsers, color: '#4A90D9', bg: 'rgba(74,144,217,0.06)' },
            { label: 'Total Books', value: '4,893', change: '+3%', icon: HiOutlineBookOpen, color: '#8B5CF6', bg: 'rgba(139,92,246,0.06)' },
            { label: 'Searches Today', value: '342', change: '+28%', icon: HiOutlineSearch, color: '#14B8A6', bg: 'rgba(20,184,166,0.06)' },
            { label: 'Active Now', value: '47', change: '', icon: HiOutlineClock, color: '#F59E0B', bg: 'rgba(245,158,11,0.06)' },
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
              <div className="flex items-start justify-between">
                <div className="w-10 h-10 rounded-[12px] flex items-center justify-center" style={{ background: stat.bg }}>
                  <stat.icon className="text-lg" style={{ color: stat.color }} />
                </div>
                {stat.change && (
                  <span className="text-xs font-bold text-bookify-teal bg-pastel-green px-2 py-0.5 rounded-full">{stat.change}</span>
                )}
              </div>
              <div className="text-2xl font-extrabold text-text-primary mt-3">{stat.value}</div>
              <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mt-0.5">{stat.label}</div>
            </motion.div>
          ))}
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* User Growth Chart */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="lg:col-span-2 rounded-[24px] p-6"
            style={{
              background: 'linear-gradient(145deg, #ffffff, #f8f6f2)',
              boxShadow: '0 8px 30px rgba(0,0,0,0.05), inset 0 1px 0 rgba(255,255,255,0.9)',
            }}
          >
            <h3 className="text-lg font-bold text-text-primary mb-6">📈 User Growth</h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={USER_DATA}>
                <defs>
                  <linearGradient id="barGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#4A90D9" />
                    <stop offset="100%" stopColor="#8B5CF6" />
                  </linearGradient>
                </defs>
                <XAxis dataKey="month" tick={{ fontSize: 12, fontWeight: 600, fill: '#8888A8' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 12, fontWeight: 600, fill: '#8888A8' }} axisLine={false} tickLine={false} />
                <Tooltip {...styledTooltip} />
                <Bar dataKey="users" fill="url(#barGrad)" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Genre Popularity */}
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
            <h3 className="text-lg font-bold text-text-primary mb-6">📊 Genre Popularity</h3>
            <ResponsiveContainer width="100%" height={180}>
              <PieChart>
                <Pie data={GENRE_POPULARITY} cx="50%" cy="50%" innerRadius={45} outerRadius={70} paddingAngle={4} dataKey="value">
                  {GENRE_POPULARITY.map((_, i) => (
                    <Cell key={i} fill={COLORS[i]} />
                  ))}
                </Pie>
                <Tooltip {...styledTooltip} />
              </PieChart>
            </ResponsiveContainer>
            <div className="space-y-2 mt-2">
              {GENRE_POPULARITY.map((entry, i) => (
                <div key={entry.name} className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full" style={{ background: COLORS[i] }} />
                  <span className="text-xs font-semibold text-text-secondary flex-1">{entry.name}</span>
                  <span className="text-xs font-bold text-text-primary">{entry.value}%</span>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Recent Searches */}
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
            <h3 className="text-lg font-bold text-text-primary mb-5">🔍 Recent Searches</h3>
            <div className="space-y-3">
              {RECENT_SEARCH.map((s, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className="flex items-center gap-3 p-3 rounded-[14px] hover:bg-surface-2 transition-colors"
                >
                  <div className="w-8 h-8 rounded-[10px] flex items-center justify-center text-sm"
                    style={{ background: 'rgba(74,144,217,0.06)' }}
                  >
                    <HiOutlineSearch className="text-bookify-blue" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <span className="text-sm font-semibold text-text-primary">{s.query}</span>
                    <span className="text-xs text-text-muted ml-2">({s.type})</span>
                  </div>
                  <span className="text-xs text-text-muted flex-shrink-0">{s.user}</span>
                  <span className="text-xs text-text-muted flex-shrink-0">{s.time}</span>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Audit Log */}
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
            <h3 className="text-lg font-bold text-text-primary mb-5">🔒 Audit Log</h3>
            <div className="space-y-3">
              {AUDIT_EVENTS.map((event, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className="flex items-start gap-3"
                >
                  <div className={`w-2 h-2 rounded-full mt-1.5 flex-shrink-0 ${
                    event.level === 'warning' ? 'bg-bookify-amber' :
                    event.level === 'success' ? 'bg-bookify-teal' :
                    'bg-bookify-blue'
                  }`} />
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-semibold text-text-primary">{event.event}</div>
                    <div className="text-xs text-text-muted">{event.user} · {event.time}</div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}

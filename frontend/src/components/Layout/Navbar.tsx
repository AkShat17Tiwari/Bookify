import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavbarScroll } from '../../hooks/useScrollReveal';
import { SignedIn, SignedOut, UserButton } from '@clerk/clerk-react';
import { HiOutlineBookOpen, HiOutlineSearch, HiOutlineHeart, HiOutlineChartBar, HiOutlineUser, HiOutlineMenu, HiOutlineX, HiOutlineSparkles, HiOutlineEmojiHappy } from 'react-icons/hi';
import { RiSparklingLine } from 'react-icons/ri';

const navLinks = [
  { path: '/', label: 'Home', icon: HiOutlineBookOpen },
  { path: '/search', label: 'Discover', icon: HiOutlineSearch },
  { path: '/multimodal', label: 'AI Fusion', icon: HiOutlineSparkles },
  { path: '/mood', label: 'Mood Reader', icon: HiOutlineEmojiHappy },
  { path: '/dashboard', label: 'Dashboard', icon: HiOutlineChartBar },
  { path: '/wishlist', label: 'Wishlist', icon: HiOutlineHeart },
  { path: '/profile', label: 'Profile', icon: HiOutlineUser },
];

export default function Navbar() {
  const { scrolled, hidden } = useNavbarScroll();
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <>
      <motion.nav
        initial={{ y: 0 }}
        animate={{ y: hidden ? -80 : 0 }}
        transition={{ duration: 0.3, ease: [0.25, 0.46, 0.45, 0.94] }}
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
          scrolled
            ? 'glass shadow-[0_4px_30px_rgba(0,0,0,0.06)]'
            : 'bg-transparent'
        }`}
        style={{ height: 72 }}
      >
        <div className="max-w-[1400px] mx-auto px-6 h-full flex items-center justify-between">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2.5 group">
            <motion.div
              whileHover={{ rotate: [0, -10, 10, -5, 0] }}
              transition={{ duration: 0.5 }}
              className="w-10 h-10 rounded-[14px] flex items-center justify-center text-xl"
              style={{
                background: 'linear-gradient(135deg, #4A90D9, #8B5CF6)',
                boxShadow: '0 4px 15px rgba(74, 144, 217, 0.3)',
              }}
            >
              📚
            </motion.div>
            <span className="text-xl font-extrabold tracking-tight">
              <span className="gradient-text">Bookify</span>
            </span>
            <RiSparklingLine className="text-bookify-purple opacity-60 group-hover:opacity-100 transition-opacity" />
          </Link>

          {/* Desktop Nav */}
          <div className="hidden md:flex items-center gap-1">
            {navLinks.map((link) => {
              const isActive = location.pathname === link.path;
              return (
                <Link key={link.path} to={link.path}>
                  <motion.div
                    whileHover={{ y: -2 }}
                    whileTap={{ scale: 0.97 }}
                    className={`relative px-5 py-2.5 rounded-[14px] text-sm font-semibold transition-all duration-300 flex items-center gap-2 ${
                      isActive
                        ? 'text-bookify-blue'
                        : 'text-text-secondary hover:text-text-primary'
                    }`}
                  >
                    <link.icon className="text-lg" />
                    {link.label}
                    {isActive && (
                      <motion.div
                        layoutId="nav-indicator"
                        className="absolute inset-0 rounded-[14px]"
                        style={{
                          background: 'linear-gradient(135deg, rgba(74,144,217,0.08), rgba(139,92,246,0.06))',
                          border: '1px solid rgba(74,144,217,0.12)',
                        }}
                        transition={{ type: 'spring', stiffness: 380, damping: 30 }}
                      />
                    )}
                  </motion.div>
                </Link>
              );
            })}
          </div>

          {/* Right Side */}
          <div className="flex items-center gap-3">
            {/* Signed Out: Show Sign In button */}
            <SignedOut>
              <Link to="/auth">
                <motion.button
                  whileHover={{ y: -2, boxShadow: '0 8px 25px rgba(74,144,217,0.2)' }}
                  whileTap={{ scale: 0.96 }}
                  className="hidden md:flex items-center gap-2 px-5 py-2.5 rounded-[14px] text-sm font-semibold text-white"
                  style={{
                    background: 'linear-gradient(135deg, #4A90D9, #8B5CF6)',
                    boxShadow: '0 4px 15px rgba(74,144,217,0.25), inset 0 1px 0 rgba(255,255,255,0.2)',
                  }}
                >
                  <HiOutlineUser className="text-lg" />
                  Sign In
                </motion.button>
              </Link>
            </SignedOut>

            {/* Signed In: Show Clerk UserButton with avatar */}
            <SignedIn>
              <UserButton
                afterSignOutUrl="/"
                appearance={{
                  elements: {
                    userButtonAvatarBox: 'w-10 h-10 ring-2 ring-white shadow-md',
                    userButtonPopoverCard: 'rounded-[18px] shadow-xl border border-gray-100',
                    userButtonPopoverActionButton: 'rounded-[12px]',
                  },
                }}
              />
            </SignedIn>

            {/* Mobile Menu Toggle */}
            <motion.button
              whileTap={{ scale: 0.9 }}
              className="md:hidden w-10 h-10 rounded-[12px] flex items-center justify-center text-text-secondary neu-raised"
              onClick={() => setMobileOpen(!mobileOpen)}
            >
              {mobileOpen ? <HiOutlineX className="text-xl" /> : <HiOutlineMenu className="text-xl" />}
            </motion.button>
          </div>
        </div>
      </motion.nav>

      {/* Mobile Menu */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="fixed top-[72px] left-0 right-0 z-40 glass p-4 md:hidden"
            style={{ borderBottom: '1px solid rgba(0,0,0,0.05)' }}
          >
            {navLinks.map((link) => (
              <Link key={link.path} to={link.path} onClick={() => setMobileOpen(false)}>
                <motion.div
                  whileTap={{ scale: 0.97 }}
                  className={`flex items-center gap-3 px-4 py-3.5 rounded-[14px] text-[15px] font-semibold mb-1 transition-all ${
                    location.pathname === link.path
                      ? 'bg-pastel-blue text-bookify-blue'
                      : 'text-text-secondary hover:bg-surface-2'
                  }`}
                >
                  <link.icon className="text-xl" />
                  {link.label}
                </motion.div>
              </Link>
            ))}

            <SignedOut>
              <Link to="/auth" onClick={() => setMobileOpen(false)}>
                <motion.div
                  whileTap={{ scale: 0.97 }}
                  className="mt-2 flex items-center justify-center gap-2 px-4 py-3.5 rounded-[14px] text-[15px] font-semibold text-white"
                  style={{ background: 'linear-gradient(135deg, #4A90D9, #8B5CF6)' }}
                >
                  <HiOutlineUser />
                  Sign In
                </motion.div>
              </Link>
            </SignedOut>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Spacer */}
      <div style={{ height: 72 }} />
    </>
  );
}

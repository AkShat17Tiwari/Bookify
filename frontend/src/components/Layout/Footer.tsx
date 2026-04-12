import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { HiOutlineHeart, HiOutlineMail } from 'react-icons/hi';
import { RiGithubLine, RiTwitterXLine } from 'react-icons/ri';

export default function Footer() {
  return (
    <footer className="relative mt-32">
      {/* Gradient top border */}
      <div className="h-[2px] bg-gradient-to-r from-transparent via-bookify-blue/30 to-transparent" />

      <div className="max-w-[1200px] mx-auto px-6 py-16">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-12">
          {/* Brand */}
          <div className="md:col-span-1">
            <Link to="/" className="flex items-center gap-2 mb-4">
              <div
                className="w-9 h-9 rounded-[12px] flex items-center justify-center text-lg"
                style={{
                  background: 'linear-gradient(135deg, #4A90D9, #8B5CF6)',
                  boxShadow: '0 4px 15px rgba(74,144,217,0.2)',
                }}
              >
                📚
              </div>
              <span className="text-lg font-extrabold gradient-text">Bookify</span>
            </Link>
            <p className="text-sm text-text-tertiary leading-relaxed">
              AI-powered book discovery platform. Find your next favorite read with intelligent recommendations.
            </p>
          </div>

          {/* Links */}
          <div>
            <h4 className="text-sm font-bold text-text-primary mb-4 tracking-wide uppercase">Platform</h4>
            <div className="flex flex-col gap-2.5">
              {['Discover', 'Dashboard', 'Wishlist', 'History'].map((l) => (
                <Link key={l} to={`/${l.toLowerCase()}`} className="text-sm text-text-tertiary hover:text-bookify-blue transition-colors">
                  {l}
                </Link>
              ))}
            </div>
          </div>

          <div>
            <h4 className="text-sm font-bold text-text-primary mb-4 tracking-wide uppercase">Features</h4>
            <div className="flex flex-col gap-2.5">
              {['AI Recommendations', 'Mood Search', 'Genre Explorer', 'Reading Analytics'].map((l) => (
                <span key={l} className="text-sm text-text-tertiary">{l}</span>
              ))}
            </div>
          </div>

          <div>
            <h4 className="text-sm font-bold text-text-primary mb-4 tracking-wide uppercase">Connect</h4>
            <div className="flex gap-3">
              {[
                { icon: RiGithubLine, href: '#' },
                { icon: RiTwitterXLine, href: '#' },
                { icon: HiOutlineMail, href: '#' },
              ].map((social, i) => (
                <motion.a
                  key={i}
                  href={social.href}
                  whileHover={{ y: -3 }}
                  whileTap={{ scale: 0.95 }}
                  className="w-10 h-10 rounded-[12px] neu-raised flex items-center justify-center text-text-tertiary hover:text-bookify-blue transition-colors"
                >
                  <social.icon className="text-lg" />
                </motion.a>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="mt-14 pt-6 border-t border-surface-3 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-xs text-text-muted">
            © 2026 Bookify. Crafted with <HiOutlineHeart className="inline text-bookify-rose" /> for book lovers.
          </p>
          <p className="text-xs text-text-muted">
            Powered by AI · Collaborative Filtering · Neural Networks
          </p>
        </div>
      </div>
    </footer>
  );
}

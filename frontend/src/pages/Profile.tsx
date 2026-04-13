import { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { useUser } from '@clerk/clerk-react';
import { staggerContainer, fadeUp } from '../lib/animations';
import SkeletonLoader from '../components/UI/SkeletonLoader';
import { useApi } from '../lib/api';
import { HiOutlineMail, HiOutlineClock, HiOutlineChartBar } from 'react-icons/hi';
import { RiSparklingFill } from 'react-icons/ri';

export default function Profile() {
  const { user: clerkUser } = useUser();
  const api = useApi();
  const apiRef = useRef(api);
  apiRef.current = api;
  const [profileData, setProfileData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const data = await apiRef.current.getProfile();
        setProfileData(data);
      } catch (err) {
        console.error('Failed to load profile:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchProfile();
  }, []);

  return (
    <div className="min-h-screen px-6 py-12">
      <div className="max-w-[1000px] mx-auto">
        {/* Header */}
        <motion.div variants={staggerContainer} initial="hidden" animate="visible" className="text-center mb-10">
          <motion.div variants={fadeUp} custom={0} className="mb-4">
             <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-xs font-bold uppercase tracking-widest"
              style={{ background: 'rgba(74,144,217,0.06)', border: '1px solid rgba(74,144,217,0.12)', color: '#4A90D9' }}>
              <RiSparklingFill /> Your Profile
            </span>
          </motion.div>
          <motion.h1 variants={fadeUp} custom={1} className="text-4xl md:text-5xl font-extrabold tracking-tight mb-4">
            Reader <span className="gradient-text">Identity</span>
          </motion.h1>
        </motion.div>

        {loading ? (
          <SkeletonLoader variant="card" count={3} />
        ) : (
          <motion.div variants={staggerContainer} initial="hidden" animate="visible" className="space-y-6">
            
            {/* User Card */}
            <motion.div variants={fadeUp} className="rounded-[22px] p-8 flex flex-col md:flex-row items-center gap-8"
              style={{
                background: 'linear-gradient(145deg, #ffffff, #f8f6f2)',
                boxShadow: '0 8px 30px rgba(0,0,0,0.05), inset 0 2px 0 rgba(255,255,255,0.9)',
                border: '1px solid rgba(0,0,0,0.03)',
              }}>
              <img 
                src={clerkUser?.imageUrl || `https://ui-avatars.com/api/?name=${clerkUser?.firstName}&background=random`} 
                alt="Profile" 
                className="w-32 h-32 rounded-full shadow-lg border-4 border-white"
              />
              <div className="text-center md:text-left">
                <h2 className="text-2xl font-extrabold mb-1">{clerkUser?.fullName}</h2>
                <p className="text-text-muted flex items-center justify-center md:justify-start gap-2 mb-4">
                  <HiOutlineMail /> {clerkUser?.primaryEmailAddress?.emailAddress}
                </p>
                <div className="flex flex-wrap items-center justify-center md:justify-start gap-4">
                  <div className="px-4 py-2 rounded-[12px] bg-bookify-blue/10 text-bookify-blue font-bold text-sm">
                    {profileData?.history?.length || 0} Books Read
                  </div>
                  <div className="px-4 py-2 rounded-[12px] bg-pink-500/10 text-pink-600 font-bold text-sm">
                    {profileData?.wishlist?.length || 0} Wishlisted
                  </div>
                  <div className="px-4 py-2 rounded-[12px] bg-amber-500/10 text-amber-600 font-bold text-sm">
                    {profileData?.ratings?.length || 0} Ratings
                  </div>
                </div>
              </div>
            </motion.div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Recent History */}
              <motion.div variants={fadeUp} className="rounded-[22px] p-6"
                 style={{
                  background: 'linear-gradient(145deg, #ffffff, #f8f6f2)',
                  boxShadow: '0 4px 15px rgba(0,0,0,0.03)',
                  border: '1px solid rgba(0,0,0,0.03)',
                }}>
                <div className="flex items-center gap-2 mb-5">
                  <HiOutlineClock className="text-xl text-bookify-blue" />
                  <h3 className="text-lg font-bold">Recent History</h3>
                </div>
                {profileData?.history?.length > 0 ? (
                  <ul className="space-y-4">
                    {profileData.history.slice(0, 5).map((item: any, i: number) => (
                      <li key={i} className="flex flex-col gap-1 pb-4 border-b border-gray-100 last:border-0">
                        <span className="font-semibold text-sm">{item.book_title || item.title}</span>
                        {item.genres && (
                          <span className="text-xs text-bookify-purple font-medium">
                            {item.genres.join(', ')}
                          </span>
                        )}
                        {(item.read_at || item.timestamp) && (
                          <span className="text-xs text-text-muted">{new Date(item.read_at || item.timestamp).toLocaleDateString()}</span>
                        )}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-text-muted">No reading history yet.</p>
                )}
              </motion.div>

              {/* Recent Ratings */}
              <motion.div variants={fadeUp} className="rounded-[22px] p-6"
                 style={{
                  background: 'linear-gradient(145deg, #ffffff, #f8f6f2)',
                  boxShadow: '0 4px 15px rgba(0,0,0,0.03)',
                  border: '1px solid rgba(0,0,0,0.03)',
                }}>
                <div className="flex items-center gap-2 mb-5">
                  <HiOutlineChartBar className="text-xl text-amber-500" />
                  <h3 className="text-lg font-bold">Recent Ratings</h3>
                </div>
                {profileData?.ratings?.length > 0 ? (
                  <ul className="space-y-4">
                    {profileData.ratings.slice(0, 5).map((item: any, i: number) => (
                      <li key={i} className="flex justify-between items-center pb-4 border-b border-gray-100 last:border-0">
                        <div className="flex flex-col gap-1">
                          <span className="font-semibold text-sm">{item.book_title}</span>
                          {(item.rated_at || item.timestamp) && (
                            <span className="text-xs text-text-muted">{new Date(item.rated_at || item.timestamp).toLocaleDateString()}</span>
                          )}
                        </div>
                        <div className="flex items-center gap-1 font-bold text-amber-500 bg-amber-50 px-2 py-1 rounded-lg text-sm">
                          ⭐ {item.rating}
                        </div>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-text-muted">No ratings yet.</p>
                )}
              </motion.div>
            </div>
            
          </motion.div>
        )}
      </div>
    </div>
  );
}

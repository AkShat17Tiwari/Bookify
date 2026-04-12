import { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { staggerContainer, fadeUp } from '../lib/animations';
import BookCard from '../components/UI/BookCard';
import SkeletonLoader from '../components/UI/SkeletonLoader';
import { useApi } from '../lib/api';
import type { Book } from '../lib/api';
import { HiOutlineCamera, HiOutlineRefresh, HiOutlineEmojiHappy } from 'react-icons/hi';
import { RiSparklingFill, RiCameraLine, RiEmotionHappyLine } from 'react-icons/ri';

const EMOTION_EMOJIS: Record<string, { emoji: string; color: string; gradient: string }> = {
  happy:     { emoji: '😊', color: '#10B981', gradient: 'from-emerald-400 to-teal-400' },
  sad:       { emoji: '😢', color: '#6366F1', gradient: 'from-indigo-400 to-purple-400' },
  angry:     { emoji: '😤', color: '#EF4444', gradient: 'from-red-400 to-orange-400' },
  surprised: { emoji: '😲', color: '#F59E0B', gradient: 'from-amber-400 to-yellow-400' },
  fearful:   { emoji: '😰', color: '#8B5CF6', gradient: 'from-violet-400 to-purple-500' },
  disgusted: { emoji: '🤢', color: '#84CC16', gradient: 'from-lime-400 to-green-400' },
  neutral:   { emoji: '😐', color: '#4A90D9', gradient: 'from-blue-400 to-cyan-400' },
};

type Step = 'intro' | 'scanning' | 'result' | 'books';

export default function MoodReader() {
  const navigate = useNavigate();
  const api = useApi();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  
  const [step, setStep] = useState<Step>('intro');
  const [cameraActive, setCameraActive] = useState(false);
  const [detectedEmotion, setDetectedEmotion] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [allExpressions, setAllExpressions] = useState<{ label: string; score: number }[]>([]);
  const [books, setBooks] = useState<Book[]>([]);
  const [genres, setGenres] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [faceApiLoaded, setFaceApiLoaded] = useState(false);
  const [mode, setMode] = useState<'classic' | 'ai'>('classic');

  // Load face-api.js models
  useEffect(() => {
    const loadModels = async () => {
      try {
        const faceapi = (window as any).faceapi;
        if (!faceapi) {
          // Dynamically load face-api.js
          const script = document.createElement('script');
          script.src = 'https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js';
          script.onload = async () => {
            const fa = (window as any).faceapi;
            const MODEL_URL = 'https://justadudewhohacks.github.io/face-api.js/models';
            await Promise.all([
              fa.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
              fa.nets.faceExpressionNet.loadFromUri(MODEL_URL),
            ]);
            setFaceApiLoaded(true);
          };
          document.head.appendChild(script);
        } else {
          const MODEL_URL = 'https://justadudewhohacks.github.io/face-api.js/models';
          await Promise.all([
            faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
            faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
          ]);
          setFaceApiLoaded(true);
        }
      } catch (e) {
        console.error('Failed to load face-api models:', e);
      }
    };
    loadModels();
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
      }
    };
  }, []);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setCameraActive(true);
      setStep('scanning');
    } catch (err) {
      console.error('Camera permission denied:', err);
      alert('Camera access is required for mood detection. Please allow camera permissions.');
    }
  }, []);

  const captureAndAnalyze = useCallback(async () => {
    if (!videoRef.current || !faceApiLoaded) return;

    const faceapi = (window as any).faceapi;
    const detections = await faceapi
      .detectAllFaces(videoRef.current, new faceapi.TinyFaceDetectorOptions())
      .withFaceExpressions();

    if (detections.length === 0) {
      alert('No face detected. Please make sure your face is clearly visible.');
      return;
    }

    const expressions = detections[0].expressions;
    const sorted = Object.entries(expressions)
      .map(([label, score]) => ({ label, score: score as number }))
      .sort((a, b) => b.score - a.score);

    const topEmotion = sorted[0].label;
    const topConfidence = Math.round(sorted[0].score * 100);

    setDetectedEmotion(topEmotion);
    setConfidence(topConfidence);
    setAllExpressions(sorted);
    setStep('result');

    // Stop camera
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      setCameraActive(false);
    }
  }, [faceApiLoaded]);

  const getRecommendations = useCallback(async () => {
    if (!detectedEmotion) return;
    setLoading(true);
    setStep('books');

    try {
      const data = await api.moodRecommend(detectedEmotion, mode);
      const mappedBooks = (data.books || []).map((b: any) => {
        if (Array.isArray(b)) {
          return { title: b[0], author: b[1], image: b[2], reasons: b[3] };
        }
        return b;
      });
      setBooks(mappedBooks);
      setGenres(data.genres || []);
    } catch (err) {
      console.error('Failed to get mood recommendations:', err);
    } finally {
      setLoading(false);
    }
  }, [detectedEmotion, mode, api]);

  const resetFlow = () => {
    setStep('intro');
    setDetectedEmotion('');
    setConfidence(0);
    setAllExpressions([]);
    setBooks([]);
    setGenres([]);
    setCameraActive(false);
  };

  const emotionInfo = EMOTION_EMOJIS[detectedEmotion] || EMOTION_EMOJIS.neutral;

  return (
    <div className="min-h-screen px-6 py-12">
      <div className="max-w-[1400px] mx-auto">
        {/* Header */}
        <motion.div variants={staggerContainer} initial="hidden" animate="visible" className="text-center mb-10">
          <motion.div variants={fadeUp} custom={0} className="mb-4">
            <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-xs font-bold uppercase tracking-widest"
              style={{ background: 'rgba(139,92,246,0.06)', border: '1px solid rgba(139,92,246,0.12)', color: '#8B5CF6' }}>
              <RiEmotionHappyLine /> AI Mood Reader
            </span>
          </motion.div>
          <motion.h1 variants={fadeUp} custom={1} className="text-4xl md:text-5xl font-extrabold tracking-tight mb-4">
            Your Face, Your <span className="gradient-text">Books</span>
          </motion.h1>
          <motion.p variants={fadeUp} custom={2} className="text-text-secondary text-lg max-w-lg mx-auto">
            Let AI read your mood through your camera and recommend the perfect books for how you feel right now
          </motion.p>
        </motion.div>

        {/* Steps Indicator */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}
          className="flex items-center justify-center gap-4 mb-12">
          {[
            { label: 'Camera', icon: '📸', active: step === 'intro' || step === 'scanning' },
            { label: 'Detect', icon: '🧠', active: step === 'result' },
            { label: 'Books', icon: '📚', active: step === 'books' },
          ].map((s, i) => (
            <div key={s.label} className="flex items-center gap-3">
              {i > 0 && <div className="w-10 h-0.5 rounded-full" style={{ background: s.active || step === 'books' && i <= 2 ? 'rgba(139,92,246,0.4)' : 'rgba(0,0,0,0.06)' }} />}
              <div className={`flex items-center gap-2 px-4 py-2.5 rounded-[14px] text-sm font-semibold transition-all ${
                s.active ? 'text-bookify-purple' : 'text-text-muted'
              }`} style={s.active ? {
                background: 'rgba(139,92,246,0.06)',
                border: '1px solid rgba(139,92,246,0.12)',
              } : { border: '1px solid rgba(0,0,0,0.04)' }}>
                <span className="text-lg">{s.icon}</span>
                {s.label}
              </div>
            </div>
          ))}
        </motion.div>

        {/* Intro / Camera */}
        <AnimatePresence mode="wait">
          {(step === 'intro' || step === 'scanning') && (
            <motion.div key="camera" initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -30 }} className="max-w-[580px] mx-auto">
              
              {/* Camera Feed */}
              <div className="rounded-[24px] overflow-hidden mb-6 relative group"
                style={{
                  background: 'linear-gradient(145deg, #f5f3ee, #fafaf8)',
                  boxShadow: `0 20px 60px rgba(0,0,0,0.06), inset 0 2px 0 rgba(255,255,255,1)${
                    cameraActive ? ', 0 0 40px rgba(139,92,246,0.1)' : ''
                  }`,
                  border: cameraActive ? '2px solid rgba(139,92,246,0.2)' : '2px solid rgba(0,0,0,0.04)',
                  aspectRatio: '4/3',
                }}>
                <video ref={videoRef} className="absolute inset-0 w-full h-full object-cover rounded-[22px]"
                  style={{ display: cameraActive ? 'block' : 'none', transform: 'scaleX(-1)' }} muted playsInline />
                <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" style={{ display: 'none' }} />
                
                {!cameraActive && (
                  <div className="flex flex-col items-center justify-center h-full gap-4 text-text-muted">
                    <motion.div animate={{ scale: [1, 1.1, 1] }} transition={{ duration: 2, repeat: Infinity }}>
                      <RiCameraLine className="text-6xl opacity-30" />
                    </motion.div>
                    <p className="text-sm font-medium">Camera preview will appear here</p>
                  </div>
                )}
                
                {cameraActive && (
                  <div className="absolute top-4 left-4 z-10">
                    <div className="flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-bold"
                      style={{ background: 'rgba(139,92,246,0.9)', color: 'white', backdropFilter: 'blur(8px)' }}>
                      <span className="w-2 h-2 rounded-full bg-white animate-pulse" />
                      LIVE
                    </div>
                  </div>
                )}
              </div>

              {/* Mode Toggle */}
              <div className="flex items-center justify-center gap-3 mb-5">
                <span className={`text-sm font-semibold ${mode === 'classic' ? 'text-bookify-blue' : 'text-text-muted'}`}>📊 Classic</span>
                <button onClick={() => setMode(mode === 'classic' ? 'ai' : 'classic')}
                  className="w-12 h-6 rounded-full relative transition-all"
                  style={{ background: mode === 'ai' ? 'rgba(139,92,246,0.3)' : 'rgba(74,144,217,0.2)' }}>
                  <motion.div animate={{ x: mode === 'ai' ? 24 : 2 }} transition={{ type: 'spring', stiffness: 500 }}
                    className="w-5 h-5 bg-white rounded-full shadow-md absolute top-0.5" />
                </button>
                <span className={`text-sm font-semibold flex items-center gap-1 ${mode === 'ai' ? 'text-bookify-purple' : 'text-text-muted'}`}>
                  <RiSparklingFill /> AI
                </span>
              </div>

              {/* Action Buttons */}
              <div className="flex gap-3 justify-center">
                {!cameraActive ? (
                  <motion.button whileHover={{ y: -3 }} whileTap={{ scale: 0.97 }}
                    onClick={startCamera}
                    disabled={!faceApiLoaded}
                    className="px-8 py-4 rounded-[16px] text-sm font-bold text-white flex items-center gap-2 disabled:opacity-40"
                    style={{
                      background: 'linear-gradient(135deg, #8B5CF6, #6366F1)',
                      boxShadow: '0 8px 25px rgba(139,92,246,0.25)',
                    }}>
                    <HiOutlineCamera className="text-lg" />
                    {faceApiLoaded ? 'Start Camera' : 'Loading AI Models...'}
                  </motion.button>
                ) : (
                  <motion.button whileHover={{ y: -3 }} whileTap={{ scale: 0.97 }}
                    onClick={captureAndAnalyze}
                    className="px-8 py-4 rounded-[16px] text-sm font-bold text-white flex items-center gap-2"
                    style={{
                      background: 'linear-gradient(135deg, #4A90D9, #14B8A6)',
                      boxShadow: '0 8px 25px rgba(74,144,217,0.25)',
                    }}>
                    <HiOutlineEmojiHappy className="text-lg" />
                    Capture & Detect Mood
                  </motion.button>
                )}
              </div>

              {!faceApiLoaded && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                  className="mt-4 text-center text-xs text-text-muted">
                  ⏳ Loading face detection models from CDN... This may take a moment.
                </motion.div>
              )}
            </motion.div>
          )}

          {/* Emotion Result */}
          {step === 'result' && (
            <motion.div key="result" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }} className="max-w-[520px] mx-auto mb-10">
              <div className="rounded-[28px] p-8 text-center"
                style={{
                  background: 'linear-gradient(145deg, #ffffff, #f8f6f2)',
                  boxShadow: '0 20px 60px rgba(0,0,0,0.06), inset 0 2px 0 rgba(255,255,255,1)',
                  border: `2px solid ${emotionInfo.color}20`,
                }}>
                <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }}
                  transition={{ type: 'spring', stiffness: 300, delay: 0.2 }}
                  className="text-7xl mb-4">
                  {emotionInfo.emoji}
                </motion.div>
                
                <h2 className="text-3xl font-extrabold capitalize mb-2"
                  style={{ color: emotionInfo.color }}>
                  {detectedEmotion}
                </h2>

                {/* Confidence */}
                <div className="mb-6">
                  <div className="text-xs font-bold uppercase tracking-widest text-text-muted mb-2">Confidence</div>
                  <div className="w-48 h-2.5 mx-auto rounded-full overflow-hidden" style={{ background: 'rgba(0,0,0,0.04)' }}>
                    <motion.div initial={{ width: 0 }} animate={{ width: `${confidence}%` }}
                      transition={{ duration: 1, ease: 'easeOut' }}
                      className="h-full rounded-full"
                      style={{ background: `linear-gradient(90deg, ${emotionInfo.color}, ${emotionInfo.color}88)` }}
                    />
                  </div>
                  <div className="text-3xl font-extrabold mt-2" style={{ color: emotionInfo.color }}>
                    {confidence}%
                  </div>
                </div>

                {/* All expressions */}
                <div className="flex flex-wrap gap-2 justify-center mb-6">
                  {allExpressions.slice(0, 5).map((expr, i) => (
                    <span key={expr.label}
                      className={`px-3 py-1.5 rounded-full text-xs font-semibold ${
                        i === 0 ? 'border-2' : ''
                      }`}
                      style={{
                        background: i === 0 ? `${emotionInfo.color}10` : 'rgba(0,0,0,0.03)',
                        borderColor: i === 0 ? `${emotionInfo.color}30` : 'transparent',
                        color: i === 0 ? emotionInfo.color : '#999',
                      }}>
                      {EMOTION_EMOJIS[expr.label]?.emoji || '😐'} {expr.label} {Math.round(expr.score * 100)}%
                    </span>
                  ))}
                </div>

                <div className="flex gap-3 justify-center">
                  <motion.button whileHover={{ y: -2 }} whileTap={{ scale: 0.97 }}
                    onClick={getRecommendations}
                    className="px-6 py-3 rounded-[14px] text-sm font-bold text-white flex items-center gap-2"
                    style={{
                      background: `linear-gradient(135deg, ${emotionInfo.color}, ${emotionInfo.color}cc)`,
                      boxShadow: `0 6px 20px ${emotionInfo.color}30`,
                    }}>
                    📚 Get Book Recommendations
                  </motion.button>
                  <motion.button whileHover={{ y: -2 }} whileTap={{ scale: 0.97 }}
                    onClick={resetFlow}
                    className="px-5 py-3 rounded-[14px] text-sm font-semibold text-text-secondary flex items-center gap-2"
                    style={{
                      background: 'linear-gradient(145deg, #fff, #f5f3ee)',
                      boxShadow: '0 2px 8px rgba(0,0,0,0.04)',
                      border: '1px solid rgba(0,0,0,0.04)',
                    }}>
                    <HiOutlineRefresh /> Retry
                  </motion.button>
                </div>
              </div>
            </motion.div>
          )}

          {/* Book Results */}
          {step === 'books' && (
            <motion.div key="books" initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -30 }}>
              
              {/* Emotion summary bar */}
              <div className="flex items-center justify-center gap-3 mb-8">
                <span className="text-3xl">{emotionInfo.emoji}</span>
                <div>
                  <p className="text-sm font-bold capitalize" style={{ color: emotionInfo.color }}>
                    {detectedEmotion} mood — {confidence}% confidence
                  </p>
                  {genres.length > 0 && (
                    <p className="text-xs text-text-muted">
                      Genres: {genres.join(', ')}
                    </p>
                  )}
                </div>
                <motion.button whileHover={{ rotate: 180 }} whileTap={{ scale: 0.9 }}
                  onClick={resetFlow}
                  className="ml-4 w-8 h-8 rounded-full flex items-center justify-center"
                  style={{ background: 'rgba(0,0,0,0.04)', border: '1px solid rgba(0,0,0,0.06)' }}>
                  <HiOutlineRefresh className="text-text-muted text-sm" />
                </motion.button>
              </div>

              {loading ? (
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-6">
                  <SkeletonLoader variant="card" count={8} />
                </div>
              ) : (
                <motion.div variants={staggerContainer} initial="hidden" animate="visible"
                  className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-6">
                  {books.map((book, i) => (
                    <BookCard key={book.title} title={book.title} author={book.author}
                      image={book.image} reasons={book.reasons} index={i}
                      onClick={() => navigate(`/book/${encodeURIComponent(book.title)}`)} />
                  ))}
                </motion.div>
              )}
              
              {!loading && books.length === 0 && (
                <div className="text-center py-16">
                  <p className="text-text-muted text-sm">No recommendations found for this mood. Try again!</p>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

import { useState, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { staggerContainer, fadeUp } from '../lib/animations';
import BookCard from '../components/UI/BookCard';
import SkeletonLoader from '../components/UI/SkeletonLoader';
import { useApi } from '../lib/api';
import type { Book } from '../lib/api';
import { HiOutlineMicrophone, HiOutlineCamera, HiOutlineClock } from 'react-icons/hi';
import { RiSparklingFill } from 'react-icons/ri';

interface ModalityState {
  text: string;
  voice: string;
  coverGenres: { name: string; score: number }[];
  emotion: string;
  historyGenres: string[];
}

export default function Multimodal() {
  const navigate = useNavigate();
  const api = useApi();
  const apiRef = useRef(api);
  apiRef.current = api;

  const [modalities, setModalities] = useState<ModalityState>({
    text: '', voice: '', coverGenres: [], emotion: '', historyGenres: [],
  });
  const [enabledModalities, setEnabledModalities] = useState({
    text: true, voice: false, cover: false, emotion: false, history: false,
  });
  const [books, setBooks] = useState<Book[]>([]);
  const [fusionGenres, setFusionGenres] = useState<string[]>([]);
  const [activeModalities, setActiveModalities] = useState(0);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const [mode, setMode] = useState<'classic' | 'ai'>('classic');

  // Voice recording state
  const [isRecording, setIsRecording] = useState(false);
  const recognitionRef = useRef<any>(null);

  // Cover URL state
  const [coverUrl, setCoverUrl] = useState('');
  const [coverAnalyzing, setCoverAnalyzing] = useState(false);
  const [coverResult, setCoverResult] = useState<{ genres: string[]; palette: string[] } | null>(null);

  const toggleModality = (key: keyof typeof enabledModalities) => {
    setEnabledModalities(prev => ({ ...prev, [key]: !prev[key] }));
  };

  // Voice search using Web Speech API
  const startVoiceRecording = useCallback(() => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert('Speech recognition is not supported in your browser. Try Chrome.');
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.onresult = (event: any) => {
      const text = event.results[0][0].transcript;
      setModalities(prev => ({ ...prev, voice: text }));
      setIsRecording(false);
    };
    recognition.onerror = () => setIsRecording(false);
    recognition.onend = () => setIsRecording(false);
    recognitionRef.current = recognition;
    recognition.start();
    setIsRecording(true);
  }, []);

  const stopVoiceRecording = useCallback(() => {
    recognitionRef.current?.stop();
    setIsRecording(false);
  }, []);

  // Cover analysis
  const analyzeCover = useCallback(async () => {
    if (!coverUrl.trim()) return;
    setCoverAnalyzing(true);
    try {
      const res = await fetch('/analyze_cover', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ image_url: coverUrl }),
      });
      const data = await res.json();
      setCoverResult({ genres: data.genres || [], palette: data.palette || [] });
      setModalities(prev => ({
        ...prev,
        coverGenres: (data.genres || []).map((g: any) => (
          typeof g === 'string' ? { name: g, score: 0.7 } : { name: g[0], score: g[1] }
        )),
      }));
    } catch (err) {
      console.error('Cover analysis failed:', err);
    } finally {
      setCoverAnalyzing(false);
    }
  }, [coverUrl]);

  // Load reading history genres
  const loadHistoryGenres = useCallback(async () => {
    try {
      const data = await apiRef.current.getHistory();
      const allGenres: Record<string, number> = {};
      (data.history || []).forEach((item: any) => {
        (item.genres || []).forEach((g: string) => {
          allGenres[g] = (allGenres[g] || 0) + 1;
        });
      });
      const topGenres = Object.entries(allGenres).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([g]) => g);
      setModalities(prev => ({ ...prev, historyGenres: topGenres }));
    } catch (err) {
      console.error('Failed to load history:', err);
    }
  }, []);

  // Fuse and recommend
  const fuseAndRecommend = useCallback(async () => {
    setLoading(true);
    setSearched(true);

    const payload: any = { mode };
    if (enabledModalities.text && modalities.text) payload.text = modalities.text;
    if (enabledModalities.voice && modalities.voice) payload.voice_text = modalities.voice;
    if (enabledModalities.cover && modalities.coverGenres.length > 0) {
      payload.image_genres = modalities.coverGenres.map(g => [g.name, g.score]);
    }
    if (enabledModalities.emotion && modalities.emotion) payload.emotion = modalities.emotion;
    if (enabledModalities.history && modalities.historyGenres.length > 0) {
      payload.history_genres = modalities.historyGenres;
    }

    try {
      const res = await fetch('/multimodal_recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      const mappedBooks = (data.books || []).map((b: any) => {
        if (Array.isArray(b)) return { title: b[0], author: b[1], image: b[2], reasons: b[3] };
        return b;
      });
      setBooks(mappedBooks);
      setFusionGenres(data.genres_used || []);
      setActiveModalities(data.modalities || 0);
    } catch (err) {
      console.error('Multimodal fusion failed:', err);
    } finally {
      setLoading(false);
    }
  }, [enabledModalities, modalities, mode]);

  const EMOTION_OPTIONS = [
    { value: 'happy', emoji: '😊', label: 'Happy' },
    { value: 'sad', emoji: '😢', label: 'Sad' },
    { value: 'neutral', emoji: '😐', label: 'Calm' },
    { value: 'surprised', emoji: '😲', label: 'Curious' },
    { value: 'fearful', emoji: '😰', label: 'Anxious' },
    { value: 'angry', emoji: '😤', label: 'Intense' },
  ];

  const enabledCount = Object.values(enabledModalities).filter(Boolean).length;

  return (
    <div className="min-h-screen px-6 py-12">
      <div className="max-w-[1400px] mx-auto">
        {/* Header */}
        <motion.div variants={staggerContainer} initial="hidden" animate="visible" className="text-center mb-10">
          <motion.div variants={fadeUp} custom={0} className="mb-4">
            <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-xs font-bold uppercase tracking-widest"
              style={{ background: 'rgba(20,184,166,0.06)', border: '1px solid rgba(20,184,166,0.12)', color: '#14B8A6' }}>
              <RiSparklingFill /> Multi-Modal AI
            </span>
          </motion.div>
          <motion.h1 variants={fadeUp} custom={1} className="text-4xl md:text-5xl font-extrabold tracking-tight mb-4">
            Smart <span className="gradient-text">Discovery</span>
          </motion.h1>
          <motion.p variants={fadeUp} custom={2} className="text-text-secondary text-lg max-w-xl mx-auto">
            Combine text search, voice, book covers, mood, and reading history for the ultimate recommendation fusion
          </motion.p>
        </motion.div>

        {/* Modality Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5 max-w-[1000px] mx-auto mb-8">
          {/* 1. Text Search */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
            className={`rounded-[22px] p-6 transition-all ${enabledModalities.text ? 'ring-2 ring-bookify-blue/20' : ''}`}
            style={{
              background: 'linear-gradient(145deg, #ffffff, #f8f6f2)',
              boxShadow: enabledModalities.text 
                ? '0 8px 30px rgba(74,144,217,0.08), inset 0 2px 0 rgba(255,255,255,1)' 
                : '0 4px 12px rgba(0,0,0,0.03), inset 0 1px 0 rgba(255,255,255,0.8)',
            }}>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <span className="text-xl">✏️</span>
                <h3 className="text-sm font-bold text-text-primary">Text Search</h3>
              </div>
              <button onClick={() => toggleModality('text')}
                className={`w-10 h-5 rounded-full relative transition-all ${enabledModalities.text ? 'bg-bookify-blue/30' : 'bg-gray-200'}`}>
                <motion.div animate={{ x: enabledModalities.text ? 20 : 2 }} className="w-4 h-4 bg-white rounded-full shadow-sm absolute top-0.5" />
              </button>
            </div>
            <p className="text-xs text-text-muted mb-3">Search by book title or genre name</p>
            <input type="text" value={modalities.text}
              onChange={(e) => setModalities(prev => ({ ...prev, text: e.target.value }))}
              disabled={!enabledModalities.text}
              placeholder="e.g., The Alchemist or Fantasy"
              className="w-full px-4 py-2.5 rounded-[12px] text-sm font-medium border-none outline-none disabled:opacity-40"
              style={{ background: 'rgba(0,0,0,0.03)' }} />
          </motion.div>

          {/* 2. Voice Search */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
            className={`rounded-[22px] p-6 transition-all ${enabledModalities.voice ? 'ring-2 ring-emerald-500/20' : ''}`}
            style={{
              background: 'linear-gradient(145deg, #ffffff, #f8f6f2)',
              boxShadow: enabledModalities.voice 
                ? '0 8px 30px rgba(16,185,129,0.08), inset 0 2px 0 rgba(255,255,255,1)' 
                : '0 4px 12px rgba(0,0,0,0.03), inset 0 1px 0 rgba(255,255,255,0.8)',
            }}>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <span className="text-xl">🎤</span>
                <h3 className="text-sm font-bold text-text-primary">Voice Search</h3>
              </div>
              <button onClick={() => toggleModality('voice')}
                className={`w-10 h-5 rounded-full relative transition-all ${enabledModalities.voice ? 'bg-emerald-500/30' : 'bg-gray-200'}`}>
                <motion.div animate={{ x: enabledModalities.voice ? 20 : 2 }} className="w-4 h-4 bg-white rounded-full shadow-sm absolute top-0.5" />
              </button>
            </div>
            <p className="text-xs text-text-muted mb-3">Speak a book title or describe what you want</p>
            {modalities.voice && (
              <p className="text-xs font-semibold text-emerald-600 mb-2 bg-emerald-50 px-3 py-1.5 rounded-lg">
                🎙️ "{modalities.voice}"
              </p>
            )}
            <motion.button whileHover={{ y: -1 }} whileTap={{ scale: 0.97 }}
              disabled={!enabledModalities.voice}
              onClick={isRecording ? stopVoiceRecording : startVoiceRecording}
              className={`w-full py-2.5 rounded-[12px] text-xs font-bold flex items-center justify-center gap-2 disabled:opacity-40 transition-all ${
                isRecording ? 'text-white' : 'text-emerald-700'
              }`}
              style={{
                background: isRecording ? 'linear-gradient(135deg, #EF4444, #DC2626)' : 'rgba(16,185,129,0.08)',
                border: `1px solid ${isRecording ? 'rgba(239,68,68,0.3)' : 'rgba(16,185,129,0.15)'}`,
              }}>
              <HiOutlineMicrophone className={isRecording ? 'animate-pulse' : ''} />
              {isRecording ? 'Stop Recording' : 'Start Recording'}
            </motion.button>
          </motion.div>

          {/* 3. Book Cover Analysis */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}
            className={`rounded-[22px] p-6 transition-all ${enabledModalities.cover ? 'ring-2 ring-pink-500/20' : ''}`}
            style={{
              background: 'linear-gradient(145deg, #ffffff, #f8f6f2)',
              boxShadow: enabledModalities.cover 
                ? '0 8px 30px rgba(236,72,153,0.08), inset 0 2px 0 rgba(255,255,255,1)' 
                : '0 4px 12px rgba(0,0,0,0.03), inset 0 1px 0 rgba(255,255,255,0.8)',
            }}>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <span className="text-xl">📷</span>
                <h3 className="text-sm font-bold text-text-primary">Cover Scanner</h3>
              </div>
              <button onClick={() => toggleModality('cover')}
                className={`w-10 h-5 rounded-full relative transition-all ${enabledModalities.cover ? 'bg-pink-500/30' : 'bg-gray-200'}`}>
                <motion.div animate={{ x: enabledModalities.cover ? 20 : 2 }} className="w-4 h-4 bg-white rounded-full shadow-sm absolute top-0.5" />
              </button>
            </div>
            <p className="text-xs text-text-muted mb-3">Paste a book cover image URL to analyze its genre</p>
            <input type="text" value={coverUrl}
              onChange={(e) => setCoverUrl(e.target.value)}
              disabled={!enabledModalities.cover}
              placeholder="Paste image URL..."
              className="w-full px-4 py-2.5 rounded-[12px] text-sm font-medium border-none outline-none disabled:opacity-40 mb-2"
              style={{ background: 'rgba(0,0,0,0.03)' }} />
            {coverResult && (
              <div className="flex flex-wrap gap-1 mb-2">
                {coverResult.genres.map((g: any) => (
                  <span key={typeof g === 'string' ? g : g[0]} className="text-[10px] px-2 py-0.5 rounded-full bg-pink-50 text-pink-600 font-semibold">
                    {typeof g === 'string' ? g : g[0]}
                  </span>
                ))}
              </div>
            )}
            <motion.button whileHover={{ y: -1 }} whileTap={{ scale: 0.97 }}
              disabled={!enabledModalities.cover || !coverUrl.trim() || coverAnalyzing}
              onClick={analyzeCover}
              className="w-full py-2.5 rounded-[12px] text-xs font-bold text-pink-700 flex items-center justify-center gap-2 disabled:opacity-40"
              style={{ background: 'rgba(236,72,153,0.08)', border: '1px solid rgba(236,72,153,0.15)' }}>
              <HiOutlineCamera />
              {coverAnalyzing ? 'Analyzing...' : 'Analyze Cover'}
            </motion.button>
          </motion.div>

          {/* 4. Emotion */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}
            className={`rounded-[22px] p-6 transition-all ${enabledModalities.emotion ? 'ring-2 ring-violet-500/20' : ''}`}
            style={{
              background: 'linear-gradient(145deg, #ffffff, #f8f6f2)',
              boxShadow: enabledModalities.emotion 
                ? '0 8px 30px rgba(139,92,246,0.08), inset 0 2px 0 rgba(255,255,255,1)' 
                : '0 4px 12px rgba(0,0,0,0.03), inset 0 1px 0 rgba(255,255,255,0.8)',
            }}>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <span className="text-xl">😊</span>
                <h3 className="text-sm font-bold text-text-primary">Mood</h3>
              </div>
              <button onClick={() => toggleModality('emotion')}
                className={`w-10 h-5 rounded-full relative transition-all ${enabledModalities.emotion ? 'bg-violet-500/30' : 'bg-gray-200'}`}>
                <motion.div animate={{ x: enabledModalities.emotion ? 20 : 2 }} className="w-4 h-4 bg-white rounded-full shadow-sm absolute top-0.5" />
              </button>
            </div>
            <p className="text-xs text-text-muted mb-3">Pick your current mood</p>
            <div className="grid grid-cols-3 gap-2">
              {EMOTION_OPTIONS.map(e => (
                <button key={e.value}
                  disabled={!enabledModalities.emotion}
                  onClick={() => setModalities(prev => ({ ...prev, emotion: e.value }))}
                  className={`py-2 rounded-[10px] text-xs font-semibold disabled:opacity-40 transition-all ${
                    modalities.emotion === e.value ? 'ring-2 ring-violet-400' : ''
                  }`}
                  style={{
                    background: modalities.emotion === e.value ? 'rgba(139,92,246,0.1)' : 'rgba(0,0,0,0.03)',
                    color: modalities.emotion === e.value ? '#7C3AED' : '#999',
                  }}>
                  {e.emoji} {e.label}
                </button>
              ))}
            </div>
          </motion.div>

          {/* 5. Reading History */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}
            className={`rounded-[22px] p-6 transition-all ${enabledModalities.history ? 'ring-2 ring-amber-500/20' : ''}`}
            style={{
              background: 'linear-gradient(145deg, #ffffff, #f8f6f2)',
              boxShadow: enabledModalities.history 
                ? '0 8px 30px rgba(245,158,11,0.08), inset 0 2px 0 rgba(255,255,255,1)' 
                : '0 4px 12px rgba(0,0,0,0.03), inset 0 1px 0 rgba(255,255,255,0.8)',
            }}>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <span className="text-xl">📖</span>
                <h3 className="text-sm font-bold text-text-primary">Reading History</h3>
              </div>
              <button onClick={() => { toggleModality('history'); if (!enabledModalities.history) loadHistoryGenres(); }}
                className={`w-10 h-5 rounded-full relative transition-all ${enabledModalities.history ? 'bg-amber-500/30' : 'bg-gray-200'}`}>
                <motion.div animate={{ x: enabledModalities.history ? 20 : 2 }} className="w-4 h-4 bg-white rounded-full shadow-sm absolute top-0.5" />
              </button>
            </div>
            <p className="text-xs text-text-muted mb-3">Use your reading history genres for recommendations</p>
            {modalities.historyGenres.length > 0 ? (
              <div className="flex flex-wrap gap-1">
                {modalities.historyGenres.map(g => (
                  <span key={g} className="text-[10px] px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 font-semibold">{g}</span>
                ))}
              </div>
            ) : (
              <div className="text-center py-3 text-xs text-text-muted">
                <HiOutlineClock className="inline mr-1" />
                {enabledModalities.history ? 'Loading history...' : 'Enable to load genres'}
              </div>
            )}
          </motion.div>

          {/* 6. Fusion CTA */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.6 }}
            className="rounded-[22px] p-6 flex flex-col items-center justify-center text-center"
            style={{
              background: 'linear-gradient(145deg, rgba(74,144,217,0.04), rgba(139,92,246,0.04))',
              border: '2px dashed rgba(74,144,217,0.15)',
            }}>
            <div className="text-4xl mb-3">🔗</div>
            <p className="text-xs text-text-muted mb-2">
              <span className="font-bold text-text-secondary">{enabledCount}</span> modalities active
            </p>
            {/* Mode Toggle */}
            <div className="flex items-center gap-2 mb-4">
              <span className={`text-xs font-semibold ${mode === 'classic' ? 'text-bookify-blue' : 'text-text-muted'}`}>Classic</span>
              <button onClick={() => setMode(mode === 'classic' ? 'ai' : 'classic')}
                className="w-10 h-5 rounded-full relative transition-all"
                style={{ background: mode === 'ai' ? 'rgba(139,92,246,0.3)' : 'rgba(74,144,217,0.2)' }}>
                <motion.div animate={{ x: mode === 'ai' ? 20 : 2 }} className="w-4 h-4 bg-white rounded-full shadow-sm absolute top-0.5" />
              </button>
              <span className={`text-xs font-semibold ${mode === 'ai' ? 'text-bookify-purple' : 'text-text-muted'}`}>AI</span>
            </div>
            <motion.button whileHover={{ y: -3, boxShadow: '0 12px 35px rgba(74,144,217,0.2)' }}
              whileTap={{ scale: 0.96 }} onClick={fuseAndRecommend}
              disabled={loading}
              className="px-8 py-3.5 rounded-[14px] text-sm font-bold text-white disabled:opacity-50"
              style={{
                background: 'linear-gradient(135deg, #4A90D9, #8B5CF6)',
                boxShadow: '0 6px 20px rgba(74,144,217,0.2)',
              }}>
              {loading ? '⏳ Fusing...' : '✨ Fuse & Recommend'}
            </motion.button>
          </motion.div>
        </div>

        {/* Fusion info bar */}
        <AnimatePresence>
          {searched && fusionGenres.length > 0 && (
            <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
              className="max-w-[800px] mx-auto mb-8 rounded-[16px] p-4 text-center"
              style={{
                background: 'rgba(74,144,217,0.04)',
                border: '1px solid rgba(74,144,217,0.12)',
              }}>
              <p className="text-sm text-text-secondary mb-2">
                🔗 Fused <strong className="text-bookify-blue">{activeModalities}</strong> modalities
              </p>
              <div className="flex flex-wrap gap-2 justify-center">
                {fusionGenres.map(g => (
                  <span key={g} className="px-3 py-1 rounded-full text-xs font-semibold"
                    style={{ background: 'rgba(74,144,217,0.08)', color: '#4A90D9', border: '1px solid rgba(74,144,217,0.15)' }}>
                    {g}
                  </span>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Results */}
        {loading && (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-6">
            <SkeletonLoader variant="card" count={8} />
          </div>
        )}

        {!loading && searched && (
          <motion.div variants={staggerContainer} initial="hidden" animate="visible"
            className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-6">
            {books.map((book, i) => (
              <BookCard key={book.title} title={book.title} author={book.author}
                image={book.image} reasons={book.reasons} index={i}
                onClick={() => navigate(`/book/${encodeURIComponent(book.title)}`)} />
            ))}
          </motion.div>
        )}

        {!loading && searched && books.length === 0 && (
          <div className="text-center py-16">
            <p className="text-text-muted">No results. Enable more modalities and try again.</p>
          </div>
        )}
      </div>
    </div>
  );
}

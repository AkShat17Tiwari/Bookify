import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { HiOutlineSearch } from 'react-icons/hi';
import { RiSparklingFill } from 'react-icons/ri';
import { useApi } from '../../lib/api';

interface SearchBarProps {
  onSearch: (query: string, mode: string) => void;
  placeholder?: string;
}

export default function SearchBar({ onSearch, placeholder = 'Search for books, genres, or authors...' }: SearchBarProps) {
  const api = useApi();
  const apiRef = useRef(api);
  apiRef.current = api;
  const [query, setQuery] = useState('');
  const [focused, setFocused] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [mode, setMode] = useState<'classic' | 'ai'>('classic');
  const inputRef = useRef<HTMLInputElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(null);

  const fetchSuggestions = useCallback(async (q: string) => {
    if (q.length < 2) {
      setSuggestions([]);
      return;
    }
    try {
      const data = await apiRef.current.autocomplete(q);
      setSuggestions(data);
    } catch {
      setSuggestions([]);
    }
  }, []);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => fetchSuggestions(query), 400);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [query, fetchSuggestions]);

  const handleSubmit = (value?: string) => {
    const q = value || query;
    if (q.trim()) {
      onSearch(q.trim(), mode);
      setSuggestions([]);
      setSelectedIndex(-1);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex((prev) => Math.min(prev + 1, suggestions.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex((prev) => Math.max(prev - 1, -1));
    } else if (e.key === 'Enter') {
      if (selectedIndex >= 0 && suggestions[selectedIndex]) {
        setQuery(suggestions[selectedIndex]);
        handleSubmit(suggestions[selectedIndex]);
      } else {
        handleSubmit();
      }
    } else if (e.key === 'Escape') {
      setSuggestions([]);
      setSelectedIndex(-1);
      inputRef.current?.blur();
    }
  };

  return (
    <div className="relative w-full max-w-2xl mx-auto">
      {/* Mode Toggle */}
      <div className="flex items-center justify-center gap-2 mb-4">
        <motion.button
          whileTap={{ scale: 0.96 }}
          onClick={() => setMode('classic')}
          className={`px-4 py-2 rounded-[12px] text-sm font-semibold transition-all duration-300 ${
            mode === 'classic'
              ? 'text-bookify-blue'
              : 'text-text-tertiary hover:text-text-secondary'
          }`}
          style={mode === 'classic' ? {
            background: 'linear-gradient(145deg, #fff, #f0edea)',
            boxShadow: '0 2px 8px rgba(74,144,217,0.12), inset 0 1px 0 rgba(255,255,255,0.9)',
            border: '1px solid rgba(74,144,217,0.15)',
          } : {}}
        >
          📊 Classic
        </motion.button>
        <motion.button
          whileTap={{ scale: 0.96 }}
          onClick={() => setMode('ai')}
          className={`px-4 py-2 rounded-[12px] text-sm font-semibold transition-all duration-300 flex items-center gap-1.5 ${
            mode === 'ai'
              ? 'text-bookify-purple'
              : 'text-text-tertiary hover:text-text-secondary'
          }`}
          style={mode === 'ai' ? {
            background: 'linear-gradient(145deg, #fff, #f0edea)',
            boxShadow: '0 2px 8px rgba(139,92,246,0.12), inset 0 1px 0 rgba(255,255,255,0.9)',
            border: '1px solid rgba(139,92,246,0.15)',
          } : {}}
        >
          <RiSparklingFill className="text-bookify-purple" />
          AI Mode
        </motion.button>
      </div>

      {/* Search Input */}
      <motion.div
        animate={{
          scale: focused ? 1.02 : 1,
          boxShadow: focused
            ? '0 8px 30px rgba(74,144,217,0.12), 0 0 0 2px rgba(74,144,217,0.15), inset 0 2px 4px rgba(0,0,0,0.03)'
            : '0 2px 8px rgba(0,0,0,0.04), inset 0 2px 4px rgba(0,0,0,0.04)',
        }}
        transition={{ duration: 0.3, ease: [0.25, 0.46, 0.45, 0.94] }}
        className="relative rounded-[18px] overflow-hidden"
        style={{
          background: 'linear-gradient(145deg, #f5f3ee, #fafaf8)',
        }}
      >
        <div className="flex items-center px-5">
          <motion.div
            animate={{ rotate: focused ? 90 : 0 }}
            transition={{ duration: 0.3 }}
          >
            <HiOutlineSearch className="text-xl text-text-tertiary" />
          </motion.div>
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => { setQuery(e.target.value); setSelectedIndex(-1); }}
            onFocus={() => setFocused(true)}
            onBlur={() => setTimeout(() => { setFocused(false); setSuggestions([]); }, 200)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            className="flex-1 bg-transparent border-none outline-none py-4.5 px-3 text-[15px] font-medium text-text-primary placeholder:text-text-muted"
          />
          {query && (
            <motion.button
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              onClick={() => { setQuery(''); setSuggestions([]); }}
              className="text-text-muted hover:text-text-secondary transition-colors text-sm"
            >
              ✕
            </motion.button>
          )}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => handleSubmit()}
            className="ml-3 px-5 py-2.5 rounded-[12px] text-sm font-bold text-white"
            style={{
              background: mode === 'ai'
                ? 'linear-gradient(135deg, #8B5CF6, #6366F1)'
                : 'linear-gradient(135deg, #4A90D9, #14B8A6)',
              boxShadow: '0 4px 12px rgba(74,144,217,0.2), inset 0 1px 0 rgba(255,255,255,0.2)',
            }}
          >
            Search
          </motion.button>
        </div>
      </motion.div>

      {/* Suggestions Dropdown */}
      <AnimatePresence>
        {suggestions.length > 0 && focused && (
          <motion.div
            initial={{ opacity: 0, y: -8, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.98 }}
            transition={{ duration: 0.2 }}
            className="absolute top-full left-0 right-0 mt-2 rounded-[16px] overflow-hidden z-50"
            style={{
              background: 'rgba(255,255,255,0.95)',
              backdropFilter: 'blur(20px)',
              boxShadow: '0 10px 40px rgba(0,0,0,0.08), 0 0 0 1px rgba(0,0,0,0.03)',
            }}
          >
            <div className="py-2 max-h-64 overflow-auto">
              {suggestions.map((suggestion, i) => (
                <motion.button
                  key={suggestion}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.03 }}
                  onClick={() => {
                    setQuery(suggestion);
                    handleSubmit(suggestion);
                  }}
                  className={`w-full text-left px-5 py-3 text-sm font-medium transition-all ${
                    i === selectedIndex
                      ? 'bg-pastel-blue text-bookify-blue'
                      : 'text-text-secondary hover:bg-surface-2'
                  }`}
                >
                  {suggestion.startsWith('📂') ? (
                    <span className="font-semibold">{suggestion}</span>
                  ) : (
                    <span>{suggestion}</span>
                  )}
                </motion.button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

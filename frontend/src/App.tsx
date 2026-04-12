import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import { ClerkProvider, SignedIn, SignedOut, RedirectToSignIn } from '@clerk/clerk-react';
import Navbar from './components/Layout/Navbar';
import Footer from './components/Layout/Footer';
import PageTransition from './components/Layout/PageTransition';
import Landing from './pages/Landing';
import Search from './pages/Search';
import BookDetail from './pages/BookDetail';
import Dashboard from './pages/Dashboard';
import Wishlist from './pages/Wishlist';
import ReadingHistory from './pages/ReadingHistory';
import MoodReader from './pages/MoodReader';
import Multimodal from './pages/Multimodal';
import Profile from './pages/Profile';
import Auth from './pages/Auth';
import Admin from './pages/Admin';

const CLERK_PUBLISHABLE_KEY = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY || 'pk_test_bW92ZWQtcHl0aG9uLTkyLmNsZXJrLmFjY291bnRzLmRldiQ';

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  return (
    <>
      <SignedIn>{children}</SignedIn>
      <SignedOut><RedirectToSignIn /></SignedOut>
    </>
  );
}

function AnimatedRoutes() {
  const location = useLocation();

  return (
    <AnimatePresence mode="wait">
      <PageTransition key={location.pathname}>
        <Routes location={location}>
          <Route path="/" element={<Landing />} />
          <Route path="/search" element={<Search />} />
          <Route path="/book/:title" element={<BookDetail />} />
          <Route path="/dashboard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
          <Route path="/wishlist" element={<ProtectedRoute><Wishlist /></ProtectedRoute>} />
          <Route path="/history" element={<ProtectedRoute><ReadingHistory /></ProtectedRoute>} />
          <Route path="/multimodal" element={<ProtectedRoute><Multimodal /></ProtectedRoute>} />
          <Route path="/mood" element={<ProtectedRoute><MoodReader /></ProtectedRoute>} />
          <Route path="/profile" element={<ProtectedRoute><Profile /></ProtectedRoute>} />
          <Route path="/auth" element={<Auth />} />
          <Route path="/admin" element={<ProtectedRoute><Admin /></ProtectedRoute>} />
        </Routes>
      </PageTransition>
    </AnimatePresence>
  );
}

export default function App() {
  return (
    <ClerkProvider publishableKey={CLERK_PUBLISHABLE_KEY}>
      <BrowserRouter>
        <div className="min-h-screen bg-surface-1">
          <Navbar />
          <main>
            <AnimatedRoutes />
          </main>
          <Footer />
        </div>
      </BrowserRouter>
    </ClerkProvider>
  );
}

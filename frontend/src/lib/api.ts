import { useAuth } from '@clerk/clerk-react';
import { useMemo } from 'react';

const API_BASE = '';

/**
 * Creates a fetch request wrapper that includes Clerk auth token.
 */
async function request<T>(url: string, options?: RequestInit, token?: string | null): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options?.headers as Record<string, string>),
  };

  // Add Clerk Bearer token if available
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE}${url}`, {
    ...options,
    headers,
    credentials: 'include',
  });

  if (!res.ok) {
    // Check if response is HTML (redirect from login_required)
    const contentType = res.headers.get('content-type') || '';
    if (contentType.includes('text/html')) {
      throw new Error('AUTH_REQUIRED');
    }
    throw new Error(`API Error: ${res.status}`);
  }

  return res.json();
}

export interface Book {
  title: string;
  author: string;
  image: string;
  votes?: number;
  rating?: number;
  reasons?: string[];
  genres?: string[];
}

export interface PopularBook {
  title: string;
  author: string;
  image: string;
  votes: number;
  rating: number;
}

export interface BookDetails {
  description: string | null;
  year: number | null;
  pages: number | null;
  subjects: string[];
}

export interface WishlistItem {
  book_title: string;
  book_author: string;
  book_image: string;
  added_at: string;
}

export interface HistoryItem {
  book_title: string;
  book_author?: string;
  book_image?: string;
  genres: string[];
  read_at?: string;
}

export interface UserProfile {
  id: string;
  name: string;
  email: string;
  role: string;
}

/**
 * Creates an API client bound to a Clerk auth token.
 */
export function createApi(getToken: () => Promise<string | null>) {
  async function authedRequest<T>(url: string, options?: RequestInit): Promise<T> {
    const token = await getToken();
    return request<T>(url, options, token);
  }

  return {
    // Popular books (no auth needed)
    getPopularBooks: () => request<{ books: PopularBook[]; stats: { total_books: number; total_genres: number; total_users: number }; model_accuracy: any }>('/api/popular'),

    // Search/autocomplete
    autocomplete: (q: string) => authedRequest<string[]>(`/autocomplete?q=${encodeURIComponent(q)}`),

    // Recommendations (uses /api/recommend which doesn't require auth)
    recommend: (title: string, mode: string = 'classic') => {
      const formData = new URLSearchParams();
      formData.append('user_input', title);
      formData.append('mode', mode);
      return request<{ data: Book[]; matched_title?: string; genre_mode?: boolean; matched_genre?: string }>('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: formData.toString(),
      });
    },

    // Book details
    getBookDetails: (title: string) => request<BookDetails>(`/book/details?title=${encodeURIComponent(title)}`),

    // Wishlist
    getWishlist: () => authedRequest<{ wishlist: WishlistItem[] }>('/wishlist'),
    addToWishlist: (title: string, author: string, image: string) =>
      authedRequest<{ status: string }>('/wishlist/add', {
        method: 'POST',
        body: JSON.stringify({ title, author, image }),
      }),
    removeFromWishlist: (title: string) =>
      authedRequest<{ status: string }>('/wishlist/remove', {
        method: 'POST',
        body: JSON.stringify({ title }),
      }),
    checkWishlist: (title: string) => authedRequest<{ wishlisted: boolean }>(`/wishlist/check?title=${encodeURIComponent(title)}`),

    // Reading history
    getHistory: () => authedRequest<{ history: HistoryItem[] }>('/history'),
    addToHistory: (title: string, author: string, image: string) =>
      authedRequest<{ status: string }>('/history', {
        method: 'POST',
        body: JSON.stringify({ title, author, image }),
      }),
    removeFromHistory: (title: string) =>
      authedRequest<{ status: string }>('/history/remove', {
        method: 'POST',
        body: JSON.stringify({ title }),
      }),

    // Rating
    rateBook: (title: string, rating: number) =>
      authedRequest<{ status: string }>('/rate', {
        method: 'POST',
        body: JSON.stringify({ title, rating }),
      }),
    checkRating: (title: string) => authedRequest<{ rating: number | null }>(`/rating/check?title=${encodeURIComponent(title)}`),

    // For You
    getForYou: () => authedRequest<{ books: Book[]; inferred_genres: string[]; has_history: boolean }>('/api/for_you'),

    // Mood
    moodRecommend: (emotion: string, mode: string = 'classic') =>
      authedRequest<{ emotion: string; genres: string[]; books: Book[] }>('/mood_recommend', {
        method: 'POST',
        body: JSON.stringify({ emotion, mode }),
      }),

    // Admin
    getAdminUsers: () => authedRequest<any[]>('/admin/users'),
    getAdminAudit: () => authedRequest<any[]>('/admin/audit'),

    // Profile
    getProfile: () => authedRequest<{ user: UserProfile; wishlist: WishlistItem[]; history: HistoryItem[]; ratings: any[] }>('/api/profile'),
  };
}

/**
 * React hook that provides an authenticated API client.
 */
export function useApi() {
  const { getToken } = useAuth();
  return useMemo(() => createApi(getToken), [getToken]);
}

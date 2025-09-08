/**
 * React hook for fetching newspaper content
 */

import { useState, useEffect } from 'react';
import api from '../lib/api';
import type { NewspaperContentResponse } from '../types/news';

interface UseNewspaperContentResult {
  content: NewspaperContentResponse | null;
  isLoading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

export function useNewspaperContent(): UseNewspaperContentResult {
  const [content, setContent] = useState<NewspaperContentResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchContent = async () => {
    try {
      setIsLoading(true);
      setError(null);

      console.log('Fetching newspaper content from API...');
      const response = await api.news.getNewspaperContent();
      console.log('Newspaper content received:', response);
      setContent(response);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch newspaper content';
      setError(errorMessage);
      console.error('Error fetching newspaper content:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchContent();
  }, []);

  return {
    content,
    isLoading,
    error,
    refetch: fetchContent
  };
}

export default useNewspaperContent;

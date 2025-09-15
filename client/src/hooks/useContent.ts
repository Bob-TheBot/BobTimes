/**
 * React hooks for content data fetching
 * Provides easy-to-use hooks for components to fetch content from the API
 */

import { useState, useEffect } from 'react';
import { 
  MainPageContent, 
  SectionPageContent, 
  StoryModel, 
  ContentSection,
  Article,
  storyToArticle
} from '@/types/content';
import { 
  contentApi, 
  getArticlesByCategory, 
  getFeaturedArticles, 
  getRecentArticles,
  getArticleBySlug
} from '@/services/contentApi';

// Generic hook for API calls with loading and error states
function useApiCall<T>(apiCall: () => Promise<T>, dependencies: any[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;

    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);
        const result = await apiCall();
        if (isMounted) {
          setData(result);
        }
      } catch (err) {
        if (isMounted) {
          setError(err instanceof Error ? err.message : 'An error occurred');
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    fetchData();

    return () => {
      isMounted = false;
    };
  }, dependencies);

  return { data, loading, error, refetch: () => setLoading(true) };
}

// Hook for main page content
export function useMainPageContent() {
  return useApiCall(() => contentApi.getMainPageContent());
}

// Hook for section page content
export function useSectionContent(section: ContentSection, page: number = 1, pageSize: number = 20) {
  return useApiCall(
    () => contentApi.getSectionContent(section, page, pageSize),
    [section, page, pageSize]
  );
}

// Hook for story by slug
export function useStoryBySlug(slug: string) {
  return useApiCall(
    () => contentApi.getStoryBySlug(slug),
    [slug]
  );
}

// Hook for popular stories
export function usePopularStories(section?: ContentSection, limit: number = 10) {
  return useApiCall(
    () => contentApi.getPopularStories(section, limit),
    [section, limit]
  );
}

// Legacy hooks for backward compatibility with existing components

// Hook for articles by category (legacy)
export function useArticlesByCategory(category: Article['category']) {
  return useApiCall(
    () => getArticlesByCategory(category),
    [category]
  );
}

// Hook for featured articles (legacy)
export function useFeaturedArticles() {
  return useApiCall(() => getFeaturedArticles());
}

// Hook for recent articles (legacy)
export function useRecentArticles(limit: number = 10) {
  return useApiCall(
    () => getRecentArticles(limit),
    [limit]
  );
}

// Hook for article by slug (legacy)
export function useArticleBySlug(slug: string) {
  return useApiCall(
    () => getArticleBySlug(slug),
    [slug]
  );
}

// Hook for multiple categories at once (for home page)
export function useHomePageData() {
  const [data, setData] = useState<{
    featuredArticles: Article[];
    recentArticles: Article[];
    newsArticles: Article[];
    sportsArticles: Article[];
    techArticles: Article[];
    economicsArticles: Article[];
    politicsArticles: Article[];
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;

    const fetchHomePageData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Fetch all data in parallel
        const [
          featured,
          recent,
          news,
          sports,
          tech,
          economics,
          politics
        ] = await Promise.all([
          getFeaturedArticles(),
          getRecentArticles(6),
          getArticlesByCategory('news'),
          getArticlesByCategory('sports'),
          getArticlesByCategory('tech'),
          getArticlesByCategory('economics'),
          getArticlesByCategory('politics')
        ]);

        if (isMounted) {
          setData({
            featuredArticles: featured,
            recentArticles: recent,
            newsArticles: news.slice(0, 3),
            sportsArticles: sports.slice(0, 3),
            techArticles: tech.slice(0, 3),
            economicsArticles: economics.slice(0, 3),
            politicsArticles: politics.slice(0, 3)
          });
        }
      } catch (err) {
        if (isMounted) {
          setError(err instanceof Error ? err.message : 'An error occurred');
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    fetchHomePageData();

    return () => {
      isMounted = false;
    };
  }, []);

  return { data, loading, error };
}

// Utility hook for handling async operations with manual trigger
export function useAsyncOperation<T>() {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const execute = async (operation: () => Promise<T>) => {
    try {
      setLoading(true);
      setError(null);
      const result = await operation();
      setData(result);
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An error occurred';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { data, loading, error, execute };
}

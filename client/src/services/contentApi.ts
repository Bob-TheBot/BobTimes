/**
 * Content API service for fetching data from the backend
 * Replaces mockData.ts with real API calls
 */

import { 
  MainPageContent, 
  SectionPageContent, 
  StoryModel, 
  ContentSection,
  ContentFilters,
  PaginationParams,
  Article,
  Author,
  storyToArticle,
  authorModelToAuthor
} from '@/types/content';

// Get API URL from configuration
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:9200';
const API_BASE = `${API_URL}/api/v1`;

/**
 * Fetch data from the API with error handling
 */
async function fetchFromApi(endpoint: string, options: RequestInit = {}) {
  try {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(options.headers as Record<string, string> || {}),
    };

    const response = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      headers,
    });

    if (!response.ok) {
      let errorMessage = `Request failed with status ${response.status}`;
      try {
        const errorData = await response.json();
        if (errorData.detail) {
          errorMessage = errorData.detail;
        } else if (errorData.message) {
          errorMessage = errorData.message;
        }
      } catch (parseError) {
        console.warn('Could not parse error response:', parseError);
      }
      throw new Error(errorMessage);
    }

    return await response.json();
  } catch (error) {
    console.error('API request failed:', error);
    throw error;
  }
}

// Content API endpoints
export const contentApi = {
  // Get main page content
  getMainPageContent: async (): Promise<MainPageContent> => {
    console.log('[contentApi] Fetching main page content from /content/main-page');
    const data = await fetchFromApi('/content/main-page');
    console.log('[contentApi] Main page data received:', data);
    return data;
  },

  // Get section page content
  getSectionContent: async (
    section: ContentSection, 
    page: number = 1, 
    pageSize: number = 20
  ): Promise<SectionPageContent> => {
    const params = new URLSearchParams({
      page: page.toString(),
      page_size: pageSize.toString()
    });
    return fetchFromApi(`/content/sections/${section}?${params}`);
  },

  // Get story by slug
  getStoryBySlug: async (slug: string): Promise<StoryModel> => {
    return fetchFromApi(`/content/stories/${slug}`);
  },

  // Search stories
  searchStories: async (
    filters: ContentFilters = {},
    pagination: PaginationParams = { page: 1, page_size: 20 }
  ): Promise<StoryModel[]> => {
    const params = new URLSearchParams({
      page: pagination.page.toString(),
      page_size: pagination.page_size.toString()
    });

    if (filters.section) {
      params.append('section', filters.section);
    }
    if (filters.search) {
      params.append('search', filters.search);
    }
    if (filters.tags && filters.tags.length > 0) {
      filters.tags.forEach(tag => params.append('tags', tag));
    }

    const endpoint = `/content/stories?${params}`;
    console.log(`[contentApi] Fetching stories from: ${endpoint}`);
    const result = await fetchFromApi(endpoint);
    console.log(`[contentApi] API response:`, result);
    return result;
  },

  // Get popular stories
  getPopularStories: async (
    section?: ContentSection, 
    limit: number = 10
  ): Promise<StoryModel[]> => {
    const params = new URLSearchParams({
      limit: limit.toString()
    });
    if (section) {
      params.append('section', section);
    }
    return fetchFromApi(`/content/popular?${params}`);
  },

  // Health check
  getHealth: async () => {
    return fetchFromApi('/content/health');
  }
};

// Legacy API functions for backward compatibility with existing components
// These convert backend responses to the old mock data format

export const getArticlesByCategory = async (category: Article['category']): Promise<Article[]> => {
  try {
    console.log(`[contentApi] getArticlesByCategory called with category: ${category}`);
    // Map frontend categories to backend sections
    const sectionMap: Record<Article['category'], ContentSection> = {
      'tech': 'technology',
      'news': 'news',
      'economics': 'economics',
      'entertainment': 'news', // Map entertainment to news for now
      'sports': 'sports',
      'politics': 'politics'
    };

    const section = sectionMap[category];
    console.log(`[contentApi] Mapped category '${category}' to section '${section}'`);
    const stories = await contentApi.searchStories({ section }, { page: 1, page_size: 50 });
    console.log(`[contentApi] Received ${stories.length} stories for section ${section}:`, stories);
    const articles = stories.map(storyToArticle);
    console.log(`[contentApi] Converted to ${articles.length} articles`);
    return articles;
  } catch (error) {
    console.error(`[contentApi] Error fetching articles for category ${category}:`, error);
    return [];
  }
};

export const getFeaturedArticles = async (): Promise<Article[]> => {
  try {
    console.log('[contentApi] getFeaturedArticles called');
    const mainContent = await contentApi.getMainPageContent();
    console.log('[contentApi] Main page content:', mainContent);
    const articles = mainContent.featured_stories.map(storyToArticle);
    console.log(`[contentApi] Converted ${articles.length} featured articles`);
    return articles;
  } catch (error) {
    console.error('[contentApi] Error fetching featured articles:', error);
    return [];
  }
};

export const getArticleById = async (id: string): Promise<Article | undefined> => {
  try {
    // Since we don't have a direct ID lookup, we'll need to search
    // In a real implementation, you might want to add an endpoint for this
    const stories = await contentApi.searchStories({}, { page: 1, page_size: 100 });
    const story = stories.find(s => s.id === id);
    return story ? storyToArticle(story) : undefined;
  } catch (error) {
    console.error(`Error fetching article ${id}:`, error);
    return undefined;
  }
};

export const getArticleBySlug = async (slug: string): Promise<Article | undefined> => {
  try {
    const story = await contentApi.getStoryBySlug(slug);
    return storyToArticle(story);
  } catch (error) {
    console.error(`Error fetching article by slug ${slug}:`, error);
    return undefined;
  }
};

export const getRecentArticles = async (limit: number = 10): Promise<Article[]> => {
  try {
    console.log(`[contentApi] getRecentArticles called with limit: ${limit}`);
    const mainContent = await contentApi.getMainPageContent();
    console.log('[contentApi] Main page content for recent:', mainContent);
    const articles = mainContent.recent_stories.slice(0, limit).map(storyToArticle);
    console.log(`[contentApi] Converted ${articles.length} recent articles`);
    return articles;
  } catch (error) {
    console.error('[contentApi] Error fetching recent articles:', error);
    return [];
  }
};

// Author functions (these will return empty for now since we don't have author endpoints)
export const getAuthorById = async (id: string): Promise<Author | undefined> => {
  // TODO: Implement when author endpoints are available
  console.warn('getAuthorById not implemented - no author endpoints available');
  return undefined;
};

export const mockAuthors: Author[] = []; // Empty for now

export default contentApi;

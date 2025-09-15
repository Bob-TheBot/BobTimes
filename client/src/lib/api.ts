// API service for making requests to the backend
import config from './config';

// Get API URL from configuration
const API_URL = config.apiUrl;



/**
 * Fetch data from the API with error handling
 */
async function fetchFromApi(endpoint: string, options: RequestInit = {}) {
  try {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(options.headers as Record<string, string> || {}),
    };

    const response = await fetch(`${API_URL}${endpoint}`, {
      ...options,
      headers,
    });

    if (!response.ok) {
      // Try to get error message from response body first
      let errorMessage = `Request failed with status ${response.status}`;
      try {
        const errorData = await response.json();
        if (errorData.detail) {
          errorMessage = errorData.detail;
        } else if (errorData.message) {
          errorMessage = errorData.message;
        }
      } catch (parseError) {
        // If we can't parse the error response, use the default message
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

// API endpoints
export const api = {
  // Health check
  getHealth: () => fetchFromApi('/health'),

  // Content endpoints
  content: {
    getMainPageContent: () => fetchFromApi('/api/v1/content/main-page'),
    getSectionContent: (section: string, page: number = 1, pageSize: number = 20) => {
      const params = new URLSearchParams({
        page: page.toString(),
        page_size: pageSize.toString()
      });
      return fetchFromApi(`/api/v1/content/sections/${section}?${params}`);
    },
    getStoryBySlug: (slug: string) => fetchFromApi(`/api/v1/content/stories/${slug}`),
    searchStories: (params: Record<string, string | number> = {}) => {
      const searchParams = new URLSearchParams();
      Object.entries(params).forEach(([key, value]) => {
        searchParams.append(key, value.toString());
      });
      return fetchFromApi(`/api/v1/content/stories?${searchParams}`);
    },
    getPopularStories: (section?: string, limit: number = 10) => {
      const params = new URLSearchParams({ limit: limit.toString() });
      if (section) params.append('section', section);
      return fetchFromApi(`/api/v1/content/popular?${params}`);
    },
    getHealthCheck: () => fetchFromApi('/api/v1/content/health'),
  },

  // News endpoints (for newspaper content)
  news: {
    getNewspaperContent: () => fetchFromApi('/api/v1/news/content'),
    getImageBase64: (imagePath: string) => fetchFromApi(`/api/v1/news/images/${imagePath}`),
  }
};

export default api;
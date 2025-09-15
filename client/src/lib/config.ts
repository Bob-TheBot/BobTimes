/**
 * Configuration service for the application
 * Loads configuration from environment variables with fallbacks
 */

interface AppConfig {
  apiUrl: string;
  // Add other configuration properties as needed
}

/**
 * Get the API URL from environment variables or use a fallback
 * In development, this will use the value from .env.development
 * In production, this will use the value from .env.production
 */
function getApiUrl(): string {
  // Use environment variable if available (injected by Vite)
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL;
  }

  // Fallback based on current environment
  if (import.meta.env.DEV) {
    return 'http://localhost:9200';
  }

  // Production fallback (could be a relative URL if API is served from same domain)
  return '/api';
}

/**
 * Application configuration
 */
const config: AppConfig = {
  apiUrl: getApiUrl(),
};

export default config;

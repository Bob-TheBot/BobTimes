# Migration from Mock Data to Backend API

This document outlines the changes made to migrate from using mock data (`mockData.ts`) to fetching real data from the backend API.

## Changes Made

### 1. New Type Definitions
- **Created**: `client/src/types/content.ts`
  - Defines types that match the backend Pydantic models
  - Includes utility functions to convert between backend and frontend types
  - Maintains backward compatibility with existing `Article` and `Author` interfaces

### 2. New API Service
- **Created**: `client/src/services/contentApi.ts`
  - Replaces `mockData.ts` with real API calls
  - Provides both new API functions and legacy compatibility functions
  - Handles error cases and API communication

### 3. React Hooks for Data Fetching
- **Created**: `client/src/hooks/useContent.ts`
  - Custom hooks for easy data fetching in components
  - Includes loading states, error handling, and data management
  - Provides both new hooks and legacy compatibility hooks

### 4. Updated Components

#### HomePage (`client/src/components/pages/HomePage.tsx`)
- Now uses `useHomePageData()` hook instead of direct mock data calls
- Added loading and error states
- Fetches data asynchronously from the backend

#### ArticlePage (`client/src/components/pages/ArticlePage.tsx`)
- Updated to use slug-based routing instead of ID-based
- Uses `useArticleBySlug()` and `useArticlesByCategory()` hooks
- Added loading and error states

#### Category Pages (e.g., `TechPage.tsx`)
- Updated to use `useArticlesByCategory()` hook
- Added loading and error states
- Handle cases where no articles are available

#### ArticleCard (`client/src/components/ui/ArticleCard.tsx`)
- Updated links to use slug-based routing (`/article/:slug` instead of `/article/:id`)
- Falls back to ID if slug is not available

### 5. Routing Updates
- **Updated**: `client/src/App.tsx`
  - Changed article route from `/article/:id` to `/article/:slug`

### 6. API Configuration
- **Updated**: `client/src/lib/api.ts`
  - Added content-specific API endpoints
  - Maintains existing API structure

## Backend API Endpoints Used

The client now connects to these backend endpoints:

- `GET /api/v1/content/main-page` - Main page content
- `GET /api/v1/content/sections/{section}` - Section-specific content
- `GET /api/v1/content/stories/{slug}` - Individual story by slug
- `GET /api/v1/content/stories` - Search/filter stories
- `GET /api/v1/content/popular` - Popular stories
- `GET /api/v1/content/health` - Health check

## Environment Configuration

Make sure your environment variables are set correctly:

```bash
# In client/.env.development
VITE_API_URL=http://localhost:9200

# In client/.env.production
VITE_API_URL=https://your-production-api-url.com
```

## Backward Compatibility

The migration maintains backward compatibility by:

1. **Legacy Functions**: All original mock data functions are still available but now fetch from the API
2. **Type Compatibility**: The `Article` and `Author` interfaces remain the same
3. **Component APIs**: Existing components don't need to change their prop interfaces

## What's Deprecated

- **`client/src/services/mockData.ts`**: This file is no longer used and can be removed
- **Mock data arrays**: `mockArticles` and `mockAuthors` are replaced with API calls

## Testing the Migration

1. **Start the backend server**: Make sure your FastAPI backend is running on `http://localhost:9200`
2. **Start the frontend**: Run `npm start` in the client directory
3. **Check the browser**: The app should now load data from the backend API
4. **Monitor network tab**: You should see API calls to `/api/v1/content/*` endpoints

## Error Handling

The new implementation includes comprehensive error handling:

- **Loading states**: Components show loading spinners while fetching data
- **Error states**: Components display error messages if API calls fail
- **Fallbacks**: Components gracefully handle missing data
- **Retry mechanisms**: Users can retry failed requests

## Performance Considerations

- **Parallel requests**: Home page fetches multiple categories in parallel
- **Caching**: Consider adding React Query or SWR for better caching
- **Pagination**: Backend supports pagination for large datasets
- **Lazy loading**: Consider implementing lazy loading for better performance

## Next Steps

1. **Remove mock data**: Delete `client/src/services/mockData.ts` once migration is confirmed working
2. **Add caching**: Consider implementing React Query for better data management
3. **Add search**: Implement search functionality using the backend search endpoints
4. **Add pagination**: Implement pagination for category pages
5. **Add real-time updates**: Consider WebSocket connections for real-time content updates

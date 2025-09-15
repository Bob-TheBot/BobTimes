/**
 * Type definitions for content API responses
 * These match the backend Pydantic models
 */

export interface ImageModel {
  id: string;
  filename: string;
  original_filename: string;
  url: string;
  alt_text: string;
  caption?: string;
  width: number;
  height: number;
  file_size: number;
  mime_type: string;
  created_at: string;
  updated_at: string;
}

export interface AuthorModel {
  id: string;
  name: string;
  email: string;
  bio?: string;
  avatar_image?: ImageModel;
  specialties: string[];
  created_at: string;
  updated_at: string;
}

export type ContentSection = 'technology' | 'economics' | 'science';

export type TechnologySubSection = 'ai_tools' | 'tech_trends' | 'quantum_computing' | 'general_tech' | 'major_deals';
export type EconomicsSubSection = 'crypto' | 'us_stock_market' | 'general_news' | 'israel_economics' | 'exits' | 'upcoming_ipos' | 'major_transactions';
export type ScienceSubSection = 'new_research' | 'biology' | 'chemistry' | 'physics';
export type ContentStatus = 'draft' | 'published' | 'archived' | 'featured';
export type Priority = 'low' | 'medium' | 'high' | 'urgent';

export interface StoryModel {
  id: string;
  title: string;
  slug: string;
  summary: string;
  content: string;
  section: ContentSection;
  sub_section?: TechnologySubSection | EconomicsSubSection | ScienceSubSection;
  status: ContentStatus;
  priority: Priority;
  author: AuthorModel;
  featured_image?: ImageModel;
  gallery_images: ImageModel[];
  tags: string[];
  read_time_minutes: number;
  view_count: number;
  published_at?: string;
  created_at: string;
  updated_at: string;
}

export interface SectionSummary {
  section: ContentSection;
  name: string;
  description: string;
  story_count: number;
  latest_update?: string;
}

export interface MainPageContent {
  breaking_news: StoryModel[];
  featured_stories: StoryModel[];
  recent_stories: StoryModel[];
  section_highlights: Record<ContentSection, StoryModel[]>;
  sections_summary: SectionSummary[];
  last_updated: string;
}

export interface SectionPageContent {
  section: ContentSection;
  section_info: SectionSummary;
  featured_stories: StoryModel[];
  recent_stories: StoryModel[];
  all_stories: StoryModel[];
  total_count: number;
  page: number;
  page_size: number;
  last_updated: string;
}

export interface ContentFilters {
  section?: ContentSection;
  search?: string;
  tags?: string[];
}

export interface PaginationParams {
  page: number;
  page_size: number;
}

// Legacy types for backward compatibility with existing components
// These map the new backend types to the old mock data structure
export interface Article {
  id: string;
  title: string;
  slug: string;
  summary: string;
  content: string;
  author: string;
  publishedAt: string;
  category: 'technology' | 'economics' | 'science';
  subCategory?: string;
  imageUrl: string;
  readTime: number;
  tags: string[];
  featured?: boolean;
}

export interface Author {
  id: string;
  name: string;
  bio: string;
  avatar: string;
  specialties: string[];
}

// Utility functions to convert between backend and frontend types
export function storyToArticle(story: StoryModel): Article {
  return {
    id: story.id,
    title: story.title,
    slug: story.slug,
    summary: story.summary,
    content: story.content,
    author: story.author.name,
    publishedAt: story.published_at || story.created_at,
    category: story.section,
    subCategory: story.sub_section,
    imageUrl: story.featured_image?.url || '',
    readTime: story.read_time_minutes,
    tags: story.tags,
    featured: story.status === 'featured'
  };
}

export function authorModelToAuthor(authorModel: AuthorModel): Author {
  return {
    id: authorModel.id,
    name: authorModel.name,
    bio: authorModel.bio || '',
    avatar: authorModel.avatar_image?.url || '',
    specialties: authorModel.specialties
  };
}

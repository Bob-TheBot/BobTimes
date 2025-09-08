/**
 * Type definitions for news API responses
 * These match the backend Pydantic models for newspaper content
 */

export interface StoryImage {
  url: string;
  local_path?: string;
  base64_data?: string;
  mime_type?: string;
  caption?: string;
  alt_text?: string;
  is_generated: boolean;
  source_type: "search" | "generated";
  file_size_kb?: number;
}

export interface StorySource {
  url: string;
  title: string;
  summary?: string;
  accessed_at: string;
}

export interface PublishedStory {
  story_id: string;
  title: string;
  content: string;
  summary: string;
  author: string;
  reporter_id: string;
  field: string;
  section: string;
  priority: string;
  sources: StorySource[];
  keywords: string[];
  images: StoryImage[];
  published_at: string;
  updated_at?: string;
  word_count: number;
  views: number;
  editorial_decision?: any; // Can be expanded if needed
}

export interface NewspaperContentResponse {
  title: string;
  tagline: string;
  stories: PublishedStory[];
  metadata: Record<string, any>;
}

export interface ImageResponse {
  image_data: string; // base64 encoded image
  mime_type: string;
  filename: string;
}

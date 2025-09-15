/**
 * Utility functions for handling images in the news application
 */

import React from 'react';
import api from '../lib/api';
import type { StoryImage, ImageResponse } from '../types/news';

/**
 * Get the image source for a story with fallback support
 * @param images - Array of story images
 * @param fallbackPath - Path to fallback image (default: '/fallback_image.jpeg')
 * @returns Promise that resolves to image source (data URL or fallback path)
 */
export async function getStoryImageSrc(
  images: StoryImage[] = [],
  fallbackPath: string = '/fallback_image.jpeg'
): Promise<string> {
  // If no images, return fallback
  if (!images || images.length === 0) {
    return fallbackPath;
  }

  // First, check if any image already has base64 data
  const imageWithBase64 = images.find(img => img.base64_data && img.mime_type);
  if (imageWithBase64?.base64_data && imageWithBase64?.mime_type) {
    return `data:${imageWithBase64.mime_type};base64,${imageWithBase64.base64_data}`;
  }

  // Try to get the first image with a local_path
  const imageWithPath = images.find(img => img.local_path);
  
  if (!imageWithPath?.local_path) {
    return fallbackPath;
  }

  try {
    // Try to fetch the image from the API
    const imageResponse: ImageResponse = await api.news.getImageBase64(imageWithPath.local_path);
    
    // Return as data URL
    return `data:${imageResponse.mime_type};base64,${imageResponse.image_data}`;
  } catch (error) {
    console.warn('Failed to load story image, using fallback:', error);
    return fallbackPath;
  }
}

/**
 * Get alt text for a story image with fallback
 * @param images - Array of story images
 * @param storyTitle - Story title to use as fallback alt text
 * @returns Alt text for the image
 */
export function getStoryImageAlt(images: StoryImage[] = [], storyTitle: string = ''): string {
  if (images && images.length > 0 && images[0].alt_text) {
    return images[0].alt_text;
  }
  
  return storyTitle ? `Image for: ${storyTitle}` : 'News story image';
}

/**
 * Get caption for a story image
 * @param images - Array of story images
 * @returns Caption text or empty string
 */
export function getStoryImageCaption(images: StoryImage[] = []): string {
  if (images && images.length > 0 && images[0].caption) {
    return images[0].caption;
  }
  
  return '';
}

/**
 * Check if a story has any images
 * @param images - Array of story images
 * @returns True if story has images with local paths
 */
export function hasStoryImages(images: StoryImage[] = []): boolean {
  return images && images.length > 0 && images.some(img => img.local_path);
}

/**
 * React hook for loading story images with fallback
 * @param images - Array of story images
 * @param fallbackPath - Path to fallback image
 * @returns Object with image source, loading state, and error state
 */
export function useStoryImage(images: StoryImage[] = [], fallbackPath: string = '/fallback_image.jpeg') {
  const [imageSrc, setImageSrc] = React.useState<string>(fallbackPath);
  const [isLoading, setIsLoading] = React.useState<boolean>(true);
  const [hasError, setHasError] = React.useState<boolean>(false);

  React.useEffect(() => {
    let isMounted = true;

    const loadImage = async () => {
      setIsLoading(true);
      setHasError(false);

      try {
        const src = await getStoryImageSrc(images, fallbackPath);
        if (isMounted) {
          setImageSrc(src);
          setHasError(src === fallbackPath && hasStoryImages(images));
        }
      } catch (error) {
        if (isMounted) {
          setImageSrc(fallbackPath);
          setHasError(true);
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    loadImage();

    return () => {
      isMounted = false;
    };
  }, [images, fallbackPath]);

  return { imageSrc, isLoading, hasError };
}

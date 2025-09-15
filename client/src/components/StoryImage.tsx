/**
 * StoryImage component for displaying story images with fallback support
 */

import React from 'react';
import { useStoryImage, getStoryImageAlt, getStoryImageCaption } from '../utils/imageUtils';
import type { StoryImage as StoryImageType } from '../types/news';

interface StoryImageProps {
  images?: StoryImageType[];
  title: string;
  className?: string;
  fallbackPath?: string;
  showCaption?: boolean;
  loading?: 'lazy' | 'eager';
}

export function StoryImage({
  images = [],
  title,
  className = '',
  fallbackPath = '/fallback_image.jpeg',
  showCaption = false,
  loading = 'lazy'
}: StoryImageProps) {
  const { imageSrc, isLoading, hasError } = useStoryImage(images, fallbackPath);
  const altText = getStoryImageAlt(images, title);
  const caption = getStoryImageCaption(images);

  return (
    <div className={`relative ${className}`}>
      {isLoading && (
        <div className="absolute inset-0 bg-gray-200 animate-pulse rounded" />
      )}
      
      <img
        src={imageSrc}
        alt={altText}
        loading={loading}
        className={`w-full h-full object-cover rounded ${isLoading ? 'opacity-0' : 'opacity-100'} transition-opacity duration-300`}
        onLoad={() => {
          // Image loaded successfully
        }}
        onError={(e) => {
          // If the image fails to load and it's not already the fallback, try fallback
          const target = e.target as HTMLImageElement;
          if (target.src !== fallbackPath) {
            target.src = fallbackPath;
          }
        }}
      />
      
      {hasError && (
        <div className="absolute top-2 right-2 bg-yellow-500 text-white text-xs px-2 py-1 rounded">
          Using fallback image
        </div>
      )}
      
      {showCaption && caption && (
        <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-50 text-white p-2 text-sm">
          {caption}
        </div>
      )}
    </div>
  );
}

/**
 * Simplified story image component for cases where you just need the image element
 */
interface SimpleStoryImageProps {
  images?: StoryImageType[];
  title: string;
  className?: string;
  fallbackPath?: string;
}

export function SimpleStoryImage({
  images = [],
  title,
  className = 'w-full h-48 object-cover',
  fallbackPath = '/fallback_image.jpeg'
}: SimpleStoryImageProps) {
  const { imageSrc } = useStoryImage(images, fallbackPath);
  const altText = getStoryImageAlt(images, title);

  return (
    <img
      src={imageSrc}
      alt={altText}
      className={className}
      onError={(e) => {
        const target = e.target as HTMLImageElement;
        if (target.src !== fallbackPath) {
          target.src = fallbackPath;
        }
      }}
    />
  );
}

export default StoryImage;

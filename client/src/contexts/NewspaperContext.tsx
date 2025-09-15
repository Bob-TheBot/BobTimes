import React, { createContext, useContext, useState, useEffect } from 'react';
import { useNewspaperContent } from '@/hooks/useNewspaperContent';
import { NewspaperContentResponse } from '@/types/news';

interface NewspaperContextType {
  content: NewspaperContentResponse | null;
  isLoading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
  getStoriesBySection: (section: string) => any[];
}

const NewspaperContext = createContext<NewspaperContextType | undefined>(undefined);

export const NewspaperProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { content, isLoading, error, refetch } = useNewspaperContent();

  // Helper function to get stories by section
  const getStoriesBySection = (section: string) => {
    if (!content) return [];
    
    // Normalize section name for comparison
    const normalizedSection = section.toLowerCase();
    
    // Map common variations
    const sectionMap: Record<string, string> = {
      'tech': 'technology',
      'news': 'national'
    };
    
    const targetSection = sectionMap[normalizedSection] || normalizedSection;
    
    return content.stories.filter(story => 
      story.section.toLowerCase() === targetSection
    );
  };

  useEffect(() => {
    if (content) {
      console.log('[NewspaperContext] Content loaded with', content.stories.length, 'stories');
    }
  }, [content]);

  return (
    <NewspaperContext.Provider value={{ content, isLoading, error, refetch, getStoriesBySection }}>
      {children}
    </NewspaperContext.Provider>
  );
};

export const useNewspaper = () => {
  const context = useContext(NewspaperContext);
  if (context === undefined) {
    throw new Error('useNewspaper must be used within a NewspaperProvider');
  }
  return context;
};
/**
 * Test component to verify newspaper API is working
 */

import React from 'react';
import { useNewspaperContent } from '../hooks/useNewspaperContent';
import { StoryImage } from '../components/StoryImage';

export function NewspaperTest() {
  const { content, isLoading, error, refetch } = useNewspaperContent();

  if (isLoading) {
    return (
      <div className="p-8">
        <h2 className="text-2xl font-bold mb-4">Loading Newspaper Content...</h2>
        <div className="animate-pulse bg-gray-200 h-32 rounded"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8">
        <h2 className="text-2xl font-bold mb-4 text-red-600">Error Loading Newspaper</h2>
        <p className="text-red-500 mb-4">{error}</p>
        <button 
          onClick={refetch}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!content) {
    return (
      <div className="p-8">
        <h2 className="text-2xl font-bold mb-4">No Content Available</h2>
        <button 
          onClick={refetch}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
        >
          Refresh
        </button>
      </div>
    );
  }

  return (
    <div className="p-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">{content.title}</h1>
        <p className="text-gray-600">{content.tagline}</p>
        <button 
          onClick={refetch}
          className="mt-2 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
        >
          Refresh Content
        </button>
      </div>

      <div className="mb-6">
        <h2 className="text-xl font-semibold mb-2">Metadata</h2>
        <pre className="bg-gray-100 p-4 rounded text-sm overflow-auto">
          {JSON.stringify(content.metadata, null, 2)}
        </pre>
      </div>

      <div>
        <h2 className="text-xl font-semibold mb-4">
          Stories ({content.stories.length})
        </h2>
        
        {content.stories.length === 0 ? (
          <p className="text-gray-500">No stories available</p>
        ) : (
          <div className="grid gap-6">
            {content.stories.map((story) => (
              <div key={story.story_id} className="border rounded-lg p-6">
                <div className="flex gap-4">
                  <div className="flex-shrink-0">
                    <StoryImage
                      images={story.images}
                      title={story.title}
                      className="w-32 h-24"
                    />
                  </div>
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold mb-2">{story.title}</h3>
                    <p className="text-gray-600 mb-2">{story.summary}</p>
                    <div className="text-sm text-gray-500 space-y-1">
                      <p><strong>Author:</strong> {story.author}</p>
                      <p><strong>Section:</strong> {story.section}</p>
                      <p><strong>Priority:</strong> {story.priority}</p>
                      <p><strong>Published:</strong> {new Date(story.published_at).toLocaleString()}</p>
                      <p><strong>Word Count:</strong> {story.word_count}</p>
                      <p><strong>Images:</strong> {story.images.length}</p>
                      <p><strong>Sources:</strong> {story.sources.length}</p>
                      {story.images.length > 0 && (
                        <div className="mt-2">
                          <p><strong>Image Details:</strong></p>
                          <ul className="ml-4 list-disc">
                            {story.images.map((img, idx) => (
                              <li key={idx} className="text-xs">
                                {img.local_path || img.url} ({img.source_type})
                                {img.file_size_kb && ` - ${img.file_size_kb}KB`}
                                {img.is_generated && ` - Generated`}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {story.sources.length > 0 && (
                        <div className="mt-2">
                          <p><strong>Sources:</strong></p>
                          <ul className="ml-4 list-disc">
                            {story.sources.map((source, idx) => (
                              <li key={idx} className="text-xs">
                                <a 
                                  href={source.url} 
                                  target="_blank" 
                                  rel="noopener noreferrer"
                                  className="text-blue-600 hover:text-blue-800 underline"
                                >
                                  {source.title}
                                </a>
                                {source.summary && ` - ${source.summary.substring(0, 100)}...`}
                                <div className="text-gray-400 text-xs">
                                  Accessed: {new Date(source.accessed_at).toLocaleString()}
                                </div>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default NewspaperTest;

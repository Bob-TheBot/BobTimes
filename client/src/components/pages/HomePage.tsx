import React, { useState } from 'react';
import { Loader2, Newspaper, X, ChevronDown, ChevronRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { useNewspaper } from '@/contexts/NewspaperContext';
import { StoryImage } from '@/components/StoryImage';

const HomePage: React.FC = () => {
  const { content, isLoading, error, refetch } = useNewspaper();
  const [selectedStory, setSelectedStory] = useState<any>(null);
  const [sourcesExpanded, setSourcesExpanded] = useState(false);

  // Log the content for debugging
  React.useEffect(() => {
    if (content) {
      console.log('[HomePage] Newspaper content loaded:', content);
      console.log('[HomePage] Number of stories:', content.stories.length);
      console.log('[HomePage] Stories by section:', 
        content.stories.reduce((acc, story) => {
          acc[story.section] = (acc[story.section] || 0) + 1;
          return acc;
        }, {} as Record<string, number>)
      );
    }
  }, [content]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p className="text-muted-foreground">Loading newspaper...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-4">Error Loading Newspaper</h1>
          <p className="text-muted-foreground mb-6">{error}</p>
          <Button onClick={refetch}>
            Try Again
          </Button>
        </div>
      </div>
    );
  }

  if (!content || content.stories.length === 0) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <Newspaper className="h-16 w-16 mx-auto mb-4 text-muted-foreground" />
          <h2 className="text-2xl font-bold mb-2">No Stories Available</h2>
          <p className="text-muted-foreground mb-6">Check back later for new content</p>
          <Button onClick={refetch}>
            Refresh
          </Button>
        </div>
      </div>
    );
  }

  // Group stories by section
  const storiesBySection = content.stories.reduce((acc, story) => {
    if (!acc[story.section]) {
      acc[story.section] = [];
    }
    acc[story.section].push(story);
    return acc;
  }, {} as Record<string, typeof content.stories>);

  // Get front page stories (high priority or explicitly marked)
  const frontPageStories = content.stories.filter(
    story => story.priority === 'breaking' || story.priority === 'high'
  );

  // Get breaking news
  const breakingNews = content.stories.filter(story => story.priority === 'breaking');

  return (
    <>
      {/* Story Modal */}
      {selectedStory && (
        <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4" onClick={() => setSelectedStory(null)}>
          <div className="bg-white dark:bg-gray-900 rounded-lg max-w-4xl max-h-[90vh] overflow-auto w-full" onClick={(e) => e.stopPropagation()}>
            <div className="sticky top-0 bg-white dark:bg-gray-900 border-b p-4 flex justify-between items-center">
              <h2 className="text-2xl font-bold">{selectedStory.title}</h2>
              <Button variant="ghost" size="sm" onClick={() => setSelectedStory(null)}>
                <X className="h-5 w-5" />
              </Button>
            </div>
            <div className="p-6">
              {selectedStory.images && selectedStory.images.length > 0 && (
                <StoryImage
                  images={selectedStory.images}
                  title={selectedStory.title}
                  className="w-full h-64 md:h-96 mb-6"
                />
              )}
              <div className="prose dark:prose-invert max-w-none">
                <div className="flex justify-between text-sm text-muted-foreground mb-4">
                  <span>By {selectedStory.author}</span>
                  <span>{new Date(selectedStory.published_at).toLocaleString()}</span>
                </div>
                <div className="mb-4">
                  <span className="text-sm font-semibold">Section: </span>
                  <span className="text-sm capitalize">{selectedStory.section}</span>
                  <span className="ml-4 text-sm font-semibold">Priority: </span>
                  <span className={`text-sm capitalize px-2 py-1 rounded ${
                    selectedStory.priority === 'breaking' ? 'bg-red-100 text-red-700' :
                    selectedStory.priority === 'high' ? 'bg-orange-100 text-orange-700' :
                    'bg-blue-100 text-blue-700'
                  }`}>{selectedStory.priority}</span>
                </div>
                <p className="text-lg font-medium mb-4 text-gray-600 dark:text-gray-400">{selectedStory.summary}</p>
                <div className="whitespace-pre-wrap">{selectedStory.content}</div>

                {/* Sources Section */}
                {selectedStory.sources && selectedStory.sources.length > 0 && (
                  <div className="mt-8 pt-6 border-t">
                    <button
                      onClick={() => setSourcesExpanded(!sourcesExpanded)}
                      className="flex items-center gap-2 text-lg font-semibold mb-4 hover:text-blue-600 transition-colors"
                    >
                      {sourcesExpanded ? (
                        <ChevronDown className="h-5 w-5" />
                      ) : (
                        <ChevronRight className="h-5 w-5" />
                      )}
                      Sources ({selectedStory.sources.length})
                    </button>
                    {sourcesExpanded && (
                      <div className="space-y-3">
                        {selectedStory.sources.map((source, idx) => (
                          <div key={idx} className="border-l-4 border-blue-200 pl-4 py-2">
                            <a
                              href={source.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-blue-600 hover:text-blue-800 underline font-medium"
                            >
                              {source.title}
                            </a>
                            {source.summary && (
                              <p className="text-sm text-gray-600 mt-1">
                                {source.summary}
                              </p>
                            )}
                            <p className="text-xs text-gray-400 mt-1">
                              Accessed: {new Date(source.accessed_at).toLocaleString()}
                            </p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                <div className="mt-6 pt-4 border-t">
                  <p className="text-sm text-muted-foreground">
                    Word count: {selectedStory.word_count} |
                    Keywords: {selectedStory.keywords?.join(', ') || 'None'}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="min-h-screen">
      {/* Hero Section */}
      <section className="bg-gradient-to-br from-primary/10 via-background to-secondary/10 py-16 md:py-24">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-4xl mx-auto">
            <h1 className="text-4xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent">
              {content.title}
            </h1>
            <p className="text-xl md:text-2xl text-muted-foreground mb-8">
              {content.tagline}
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button size="lg" onClick={refetch}>
                <Newspaper className="mr-2 h-4 w-4" />
                Refresh News
              </Button>
              {content.metadata && (
                <div className="text-sm text-muted-foreground mt-2">
                  <p>Status: {content.metadata.status}</p>
                  <p>Total Stories: {content.metadata.total_stories}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* Breaking News Section - if any */}
      {breakingNews.length > 0 && (
        <>
          <section className="py-16 bg-red-50/30 dark:bg-red-950/10">
            <div className="container mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex items-center justify-between mb-8">
                <h2 className="text-3xl font-bold flex items-center">
                  <span className="text-red-600">Breaking News</span>
                </h2>
              </div>
              <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                {breakingNews.map((story) => (
                  <Card 
                    key={story.story_id} 
                    className="border-red-200 dark:border-red-900 cursor-pointer hover:shadow-xl transition-shadow"
                    onClick={() => {
                      setSelectedStory(story);
                      setSourcesExpanded(false);
                    }}
                  >
                    <CardHeader>
                      <StoryImage
                        images={story.images}
                        title={story.title}
                        className="w-full h-48 mb-4"
                      />
                      <CardTitle className="line-clamp-2">{story.title}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-muted-foreground line-clamp-3 mb-4">
                        {story.summary}
                      </p>
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>{story.author}</span>
                        <span>{new Date(story.published_at).toLocaleDateString()}</span>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          </section>
          <Separator />
        </>
      )}

      {/* Front Page Stories */}
      {frontPageStories.length > 0 && (
        <>
          <section className="py-16">
            <div className="container mx-auto px-4 sm:px-6 lg:px-8">
              <h2 className="text-3xl font-bold mb-8">Top Stories</h2>
              <div className="grid gap-6 md:grid-cols-2">
                {frontPageStories.slice(0, 4).map((story) => (
                  <Card 
                    key={story.story_id}
                    className="cursor-pointer hover:shadow-xl transition-shadow"
                    onClick={() => {
                      setSelectedStory(story);
                      setSourcesExpanded(false);
                    }}
                  >
                    <CardHeader>
                      <StoryImage
                        images={story.images}
                        title={story.title}
                        className="w-full h-56 mb-4"
                      />
                      <CardTitle>{story.title}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-muted-foreground mb-4">
                        {story.summary}
                      </p>
                      <div className="flex justify-between items-center text-sm">
                        <span className="text-muted-foreground">
                          By {story.author} | {story.section}
                        </span>
                        <span className="text-xs text-muted-foreground">
                          {story.word_count} words
                        </span>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          </section>
          <Separator />
        </>
      )}

      {/* Stories by Section */}
      <section className="py-16 bg-muted/30">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-center mb-12">Stories by Section</h2>
          
          <div className="space-y-12">
            {Object.entries(storiesBySection).map(([section, sectionStories]) => (
              <div key={section}>
                <h3 className="text-2xl font-semibold mb-6 capitalize">
                  {section} ({sectionStories.length})
                </h3>
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                  {sectionStories.slice(0, 6).map((story) => (
                    <Card 
                      key={story.story_id} 
                      className="cursor-pointer hover:shadow-lg transition-shadow"
                      onClick={() => {
                        setSelectedStory(story);
                        setSourcesExpanded(false);
                      }}
                    >
                      <CardHeader className="pb-3">
                        {story.images && story.images.length > 0 && (
                          <StoryImage
                            images={story.images}
                            title={story.title}
                            className="w-full h-40 mb-3"
                          />
                        )}
                        <CardTitle className="text-lg line-clamp-2">
                          {story.title}
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <p className="text-sm text-muted-foreground line-clamp-2 mb-3">
                          {story.summary}
                        </p>
                        <div className="flex justify-between items-center text-xs text-muted-foreground">
                          <span>{story.author}</span>
                          <span className={`px-2 py-1 rounded-full ${
                            story.priority === 'breaking' ? 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300' :
                            story.priority === 'high' ? 'bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-300' :
                            story.priority === 'medium' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300' :
                            'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'
                          }`}>
                            {story.priority}
                          </span>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>


    </div>
    </>
  );
};

export default HomePage;
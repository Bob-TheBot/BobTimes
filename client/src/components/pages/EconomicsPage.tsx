import React, { useState } from 'react';
import { TrendingUp, DollarSign, Globe, BarChart3, Loader2, X } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { useNewspaper } from '@/contexts/NewspaperContext';
import { StoryImage } from '@/components/StoryImage';

const EconomicsPage: React.FC = () => {
  const { isLoading, error, getStoriesBySection } = useNewspaper();
  const [selectedStory, setSelectedStory] = useState<any>(null);
  
  const economicsArticles = getStoriesBySection('economics');

  const economicsTopics = [
    { name: 'Markets', icon: TrendingUp },
    { name: 'Finance', icon: DollarSign },
    { name: 'Global Economy', icon: Globe },
    { name: 'Analytics', icon: BarChart3 },
  ];

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p className="text-muted-foreground">Loading economics articles...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-4">Error Loading Articles</h1>
          <p className="text-muted-foreground mb-6">{error}</p>
        </div>
      </div>
    );
  }

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
                <p className="text-lg font-medium mb-4 text-gray-600 dark:text-gray-400">{selectedStory.summary}</p>
                <div className="whitespace-pre-wrap">{selectedStory.content}</div>
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
        <section className="bg-gradient-to-br from-amber-50 via-background to-amber-50/30 py-16 md:py-24 dark:from-amber-950/20 dark:to-amber-950/10">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center max-w-4xl mx-auto">
              <h1 className="text-4xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-amber-600 to-amber-500 bg-clip-text text-transparent">
                Economics
              </h1>
              <p className="text-xl md:text-2xl text-muted-foreground mb-8">
                Expert analysis of financial markets, economic trends, and developments shaping the global economy.
              </p>
              
              {/* Topic badges */}
              <div className="flex flex-wrap justify-center gap-3">
                {economicsTopics.map(({ name, icon: Icon }) => (
                  <Badge
                    key={name}
                    variant="secondary"
                    className="text-sm py-2 px-4 flex items-center gap-2"
                  >
                    <Icon className="h-4 w-4" />
                    {name}
                  </Badge>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* Articles Section */}
        <section className="py-16">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8">
            <div className="mb-8">
              <h2 className="text-2xl font-bold">Economics Stories ({economicsArticles.length})</h2>
              <p className="text-muted-foreground mt-2">
                {economicsArticles.length > 0 ? 'Latest market analysis and economic insights' : 'Check back later for economics coverage'}
              </p>
            </div>

            {economicsArticles.length === 0 ? (
              <div className="text-center py-12 bg-muted/30 rounded-lg">
                <BarChart3 className="h-16 w-16 mx-auto mb-4 text-muted-foreground" />
                <p className="text-lg text-muted-foreground mb-2">No economics articles available at the moment</p>
                <p className="text-sm text-muted-foreground">Our economics coverage will be updated soon</p>
              </div>
            ) : (
              <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                {economicsArticles.map((story) => (
                  <Card 
                    key={story.story_id}
                    className="cursor-pointer hover:shadow-lg transition-shadow"
                    onClick={() => setSelectedStory(story)}
                  >
                    <CardHeader>
                      {story.images && story.images.length > 0 && (
                        <StoryImage
                          images={story.images}
                          title={story.title}
                          className="w-full h-48 mb-4"
                        />
                      )}
                      <CardTitle className="line-clamp-2">{story.title}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-muted-foreground line-clamp-3 mb-4">
                        {story.summary}
                      </p>
                      <div className="flex justify-between items-center text-xs text-muted-foreground">
                        <span>{story.author}</span>
                        <span className={`px-2 py-1 rounded-full ${
                          story.priority === 'breaking' ? 'bg-red-100 text-red-700' :
                          story.priority === 'high' ? 'bg-orange-100 text-orange-700' :
                          'bg-blue-100 text-blue-700'
                        }`}>
                          {story.priority}
                        </span>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </div>
        </section>
      </div>
    </>
  );
};

export default EconomicsPage;
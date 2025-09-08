import React, { useState } from 'react';
import { Cpu, Smartphone, Globe, Zap, Loader2, X } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { useNewspaper } from '@/contexts/NewspaperContext';
import { StoryImage } from '@/components/StoryImage';

const TechPage: React.FC = () => {
  const { content, isLoading, error, getStoriesBySection } = useNewspaper();
  const [selectedStory, setSelectedStory] = useState<any>(null);
  
  const techArticles = getStoriesBySection('technology');

  const techTopics = [
    { name: 'Artificial Intelligence', icon: Cpu },
    { name: 'Quantum Computing', icon: Smartphone },
    { name: 'Cloud Technology', icon: Globe },
    { name: 'Innovation', icon: Zap },
  ];

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p className="text-muted-foreground">Loading technology articles...</p>
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
        {/* Header Section */}
        <section className="bg-gradient-to-r from-blue-600/10 to-cyan-600/10 py-16">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center max-w-3xl mx-auto">
              <h1 className="text-4xl md:text-5xl font-bold mb-4 text-blue-600">
                Technology
              </h1>
              <p className="text-xl text-muted-foreground mb-8">
                In-depth analysis of technological developments, innovation trends, and industry transformation.
              </p>
              
              {/* Topic badges */}
              <div className="flex flex-wrap justify-center gap-3">
                {techTopics.map(({ name, icon: Icon }) => (
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
              <h2 className="text-2xl font-bold">Technology Stories ({techArticles.length})</h2>
              <p className="text-muted-foreground mt-2">
                Latest updates from the world of technology
              </p>
            </div>

            {techArticles.length === 0 ? (
              <div className="text-center py-12">
                <p className="text-muted-foreground">No technology articles available at the moment.</p>
              </div>
            ) : (
              <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                {techArticles.map((story) => (
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

export default TechPage;
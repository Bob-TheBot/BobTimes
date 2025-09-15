import React from 'react';
import { Film, Music, Star, Tv, Loader2 } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import ArticleList from '@/components/ui/ArticleList';
import { useArticlesByCategory } from '@/hooks/useContent';

const EntertainmentPage: React.FC = () => {
  const { data: entertainmentArticles, loading, error } = useArticlesByCategory('entertainment');

  const entertainmentTopics = [
    { name: 'Movies & Film', icon: Film, count: 16 },
    { name: 'Music Industry', icon: Music, count: 12 },
    { name: 'Celebrity News', icon: Star, count: 20 },
    { name: 'TV & Streaming', icon: Tv, count: 14 },
  ];

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p className="text-muted-foreground">Loading entertainment articles...</p>
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
    <div className="min-h-screen">
      {/* Header Section */}
      <section className="bg-gradient-to-r from-purple-600/10 to-pink-600/10 py-16">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-3xl mx-auto">
            <h1 className="text-4xl md:text-5xl font-bold mb-4 text-purple-600">
              Entertainment
            </h1>
            <p className="text-xl text-muted-foreground mb-8">
              Dive into the world of entertainment with the latest on movies, music, celebrities, and pop culture.
            </p>
            
            {/* Topic Tags */}
            <div className="flex flex-wrap justify-center gap-3">
              {entertainmentTopics.map((topic) => {
                const IconComponent = topic.icon;
                return (
                  <Badge 
                    key={topic.name} 
                    variant="secondary" 
                    className="px-4 py-2 text-sm bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-300"
                  >
                    <IconComponent className="w-4 h-4 mr-2" />
                    {topic.name} ({topic.count})
                  </Badge>
                );
              })}
            </div>
          </div>
        </div>
      </section>

      {/* Articles Section */}
      <section className="py-16">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between mb-8">
            <h2 className="text-2xl font-bold">Entertainment News</h2>
            <p className="text-muted-foreground">
              {entertainmentArticles?.length || 0} article{(entertainmentArticles?.length || 0) !== 1 ? 's' : ''}
            </p>
          </div>

          {entertainmentArticles && entertainmentArticles.length > 0 ? (
            <ArticleList articles={entertainmentArticles} variant="grid" />
          ) : (
            <div className="text-center py-12">
              <Star className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-xl font-semibold mb-2">No Entertainment Articles Yet</h3>
              <p className="text-muted-foreground">
                Check back soon for the latest entertainment news and celebrity updates.
              </p>
            </div>
          )}
        </div>
      </section>

      {/* Entertainment Coverage Section */}
      <section className="py-16 bg-muted/30">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-3xl mx-auto">
            <h2 className="text-3xl font-bold mb-4">Entertainment Universe</h2>
            <p className="text-lg text-muted-foreground mb-8">
              From Hollywood blockbusters to chart-topping hits, we cover the entertainment industry 
              with insider access and exclusive interviews.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-12">
              {entertainmentTopics.map((topic) => {
                const IconComponent = topic.icon;
                return (
                  <div key={topic.name} className="text-center p-6 rounded-lg bg-background border">
                    <IconComponent className="w-12 h-12 mx-auto text-purple-600 mb-4" />
                    <h3 className="font-semibold mb-2">{topic.name}</h3>
                    <p className="text-sm text-muted-foreground">
                      {topic.count} articles on entertainment
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default EntertainmentPage;

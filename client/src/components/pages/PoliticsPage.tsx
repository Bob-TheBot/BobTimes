import React from 'react';
import { Vote, Globe, Building, Scale, Loader2 } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import ArticleList from '@/components/ui/ArticleList';
import { useArticlesByCategory } from '@/hooks/useContent';

const PoliticsPage: React.FC = () => {
  const { data: politicsArticles, loading, error } = useArticlesByCategory('politics');

  const politicsTopics = [
    { name: 'Elections', icon: Vote, count: 10 },
    { name: 'International Relations', icon: Globe, count: 14 },
    { name: 'Government Policy', icon: Building, count: 12 },
    { name: 'Legislative Affairs', icon: Scale, count: 8 },
  ];

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p className="text-muted-foreground">Loading politics articles...</p>
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
      {/* Hero Section */}
      <section className="bg-gradient-to-br from-slate-50 via-background to-slate-50/30 py-16 md:py-24 dark:from-slate-950/20 dark:to-slate-950/10">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-4xl mx-auto">
            <h1 className="text-4xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-slate-600 to-slate-500 bg-clip-text text-transparent">
              Political Coverage
            </h1>
            <p className="text-xl md:text-2xl text-muted-foreground mb-8">
              In-depth analysis of government affairs, policy developments, and democratic processes.
            </p>
            
            <div className="flex flex-wrap justify-center gap-2 mb-8">
              {politicsTopics.map((topic) => (
                <Badge key={topic.name} variant="secondary" className="text-sm">
                  {topic.name}
                </Badge>
              ))}
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-12">
              {politicsTopics.map((topic) => {
                const IconComponent = topic.icon;
                return (
                  <div key={topic.name} className="text-center p-6 rounded-lg bg-background border">
                    <IconComponent className="w-12 h-12 mx-auto text-slate-600 mb-4" />
                    <h3 className="font-semibold mb-2">{topic.name}</h3>
                    <p className="text-sm text-muted-foreground">
                      {topic.count} articles on political developments
                    </p>
                  </div>
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
            <h2 className="text-2xl font-bold">Political Analysis</h2>
            <p className="text-muted-foreground">
              {politicsArticles?.length || 0} article{(politicsArticles?.length || 0) !== 1 ? 's' : ''}
            </p>
          </div>

          {politicsArticles && politicsArticles.length > 0 ? (
            <ArticleList articles={politicsArticles} variant="grid" />
          ) : (
            <div className="text-center py-12">
              <Vote className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-xl font-semibold mb-2">No Political Articles Yet</h3>
              <p className="text-muted-foreground">
                Check back soon for the latest political analysis and government coverage.
              </p>
            </div>
          )}
        </div>
      </section>
    </div>
  );
};

export default PoliticsPage;

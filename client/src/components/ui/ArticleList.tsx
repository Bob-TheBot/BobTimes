import React from 'react';
import ArticleCard from './ArticleCard';
import { Article } from '@/types/content';
import { cn } from '@/lib/utils';

interface ArticleListProps {
  articles: Article[];
  variant?: 'grid' | 'list' | 'featured';
  className?: string;
  showFeatured?: boolean;
}

const ArticleList: React.FC<ArticleListProps> = ({ 
  articles, 
  variant = 'grid',
  className,
  showFeatured = false 
}) => {
  if (articles.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-muted-foreground">No articles found.</p>
      </div>
    );
  }

  if (variant === 'featured') {
    const featuredArticle = articles.find(article => article.featured) || articles[0];
    const otherArticles = articles.filter(article => article.id !== featuredArticle.id).slice(0, 4);

    return (
      <div className={cn("space-y-8", className)}>
        {/* Featured Article */}
        <div className="mb-8">
          <ArticleCard article={featuredArticle} variant="featured" />
        </div>

        {/* Other Articles Grid */}
        {otherArticles.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {otherArticles.map((article) => (
              <ArticleCard 
                key={article.id} 
                article={article} 
                variant="default"
              />
            ))}
          </div>
        )}
      </div>
    );
  }

  if (variant === 'list') {
    return (
      <div className={cn("space-y-4", className)}>
        {articles.map((article) => (
          <ArticleCard 
            key={article.id} 
            article={article} 
            variant="compact"
          />
        ))}
      </div>
    );
  }

  // Default grid variant
  return (
    <div className={cn(
      "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6",
      className
    )}>
      {articles.map((article) => (
        <ArticleCard 
          key={article.id} 
          article={article} 
          variant="default"
        />
      ))}
    </div>
  );
};

export default ArticleList;

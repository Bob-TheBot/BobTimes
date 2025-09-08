import React from 'react';
import { Link } from 'react-router-dom';
import { Clock, User } from 'lucide-react';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { Article } from '@/types/content';

interface ArticleCardProps {
  article: Article;
  variant?: 'default' | 'featured' | 'compact';
  className?: string;
}

const ArticleCard: React.FC<ArticleCardProps> = ({ 
  article, 
  variant = 'default',
  className 
}) => {
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  const getCategoryColor = (category: string) => {
    const colors = {
      tech: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300',
      news: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300',
      economics: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300',
      entertainment: 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-300'
    };
    return colors[category as keyof typeof colors] || 'bg-gray-100 text-gray-800';
  };

  if (variant === 'compact') {
    return (
      <Link to={`/article/${article.slug || article.id}`} className="block">
        <Card className={cn("hover:shadow-md transition-shadow", className)}>
          <CardContent className="p-4">
            <div className="flex space-x-4">
              <img
                src={article.imageUrl}
                alt={article.title}
                className="w-20 h-20 object-cover rounded-md flex-shrink-0"
              />
              <div className="flex-1 min-w-0">
                <Badge className={cn("mb-2", getCategoryColor(article.category))}>
                  {article.category.charAt(0).toUpperCase() + article.category.slice(1)}
                </Badge>
                <h3 className="font-semibold text-sm line-clamp-2 mb-2">
                  {article.title}
                </h3>
                <div className="flex items-center text-xs text-muted-foreground space-x-3">
                  <span className="flex items-center">
                    <User className="w-3 h-3 mr-1" />
                    {article.author}
                  </span>
                  <span className="flex items-center">
                    <Clock className="w-3 h-3 mr-1" />
                    {article.readTime} min
                  </span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </Link>
    );
  }

  if (variant === 'featured') {
    return (
      <Link to={`/article/${article.slug || article.id}`} className="block">
        <Card className={cn("hover:shadow-lg transition-shadow", className)}>
          <div className="relative">
            <img
              src={article.imageUrl}
              alt={article.title}
              className="w-full h-64 md:h-80 object-cover rounded-t-lg"
            />
            <div className="absolute top-4 left-4">
              <Badge className={cn("text-white", getCategoryColor(article.category))}>
                {article.category.charAt(0).toUpperCase() + article.category.slice(1)}
              </Badge>
            </div>
          </div>
          <CardContent className="p-6">
            <h2 className="text-2xl font-bold mb-3 line-clamp-2">
              {article.title}
            </h2>
            <p className="text-muted-foreground mb-4 line-clamp-3">
              {article.summary}
            </p>
            <div className="flex items-center justify-between text-sm text-muted-foreground">
              <div className="flex items-center space-x-4">
                <span className="flex items-center">
                  <User className="w-4 h-4 mr-1" />
                  {article.author}
                </span>
                <span className="flex items-center">
                  <Clock className="w-4 h-4 mr-1" />
                  {article.readTime} min read
                </span>
              </div>
              <span>{formatDate(article.publishedAt)}</span>
            </div>
          </CardContent>
        </Card>
      </Link>
    );
  }

  // Default variant
  return (
    <Link to={`/article/${article.slug || article.id}`} className="block">
      <Card className={cn("hover:shadow-md transition-shadow", className)}>
        <div className="relative">
          <img
            src={article.imageUrl}
            alt={article.title}
            className="w-full h-48 object-cover rounded-t-lg"
          />
          <div className="absolute top-3 left-3">
            <Badge className={getCategoryColor(article.category)}>
              {article.category.charAt(0).toUpperCase() + article.category.slice(1)}
            </Badge>
          </div>
        </div>
        <CardContent className="p-4">
          <h3 className="font-semibold text-lg mb-2 line-clamp-2">
            {article.title}
          </h3>
          <p className="text-muted-foreground text-sm mb-3 line-clamp-2">
            {article.summary}
          </p>
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <div className="flex items-center space-x-3">
              <span className="flex items-center">
                <User className="w-3 h-3 mr-1" />
                {article.author}
              </span>
              <span className="flex items-center">
                <Clock className="w-3 h-3 mr-1" />
                {article.readTime} min
              </span>
            </div>
            <span>{formatDate(article.publishedAt)}</span>
          </div>
        </CardContent>
      </Card>
    </Link>
  );
};

export default ArticleCard;

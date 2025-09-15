import React from 'react';
import { useTheme } from '@/contexts/ThemeContext';

interface LogoProps {
  className?: string;
  width?: number;
  height?: number;
  variant?: 'default' | 'light' | 'dark' | 'icon-only';
}

export const Logo: React.FC<LogoProps> = ({ 
  className = '', 
  width = 200, 
  height = 60,
  variant = 'default'
}) => {
  const { theme } = useTheme();

  const getLogoSrc = () => {
    if (variant === 'icon-only') {
      return '/icon.svg';
    }
    
    if (variant === 'light') {
      return '/logo-light.svg';
    }
    
    if (variant === 'dark') {
      return '/logo-dark.svg';
    }
    
    // Default: use theme-aware logo
    if (theme === 'dark') {
      return '/logo-dark.svg';
    } else if (theme === 'light') {
      return '/logo-light.svg';
    } else {
      // System theme - use CSS to handle theme switching
      return '/logo.svg';
    }
  };

  const logoSrc = getLogoSrc();
  const isIconOnly = variant === 'icon-only';
  const logoWidth = isIconOnly ? 64 : width;
  const logoHeight = isIconOnly ? 64 : height;

  return (
    <img
      src={logoSrc}
      alt="Bob Times Logo"
      width={logoWidth}
      height={logoHeight}
      className={`${className} ${isIconOnly ? 'rounded-lg' : ''}`}
      style={{
        maxWidth: '100%',
        height: 'auto'
      }}
    />
  );
};

// Alternative component for when you want to use it as a link
interface LogoLinkProps extends LogoProps {
  href?: string;
  onClick?: () => void;
}

export const LogoLink: React.FC<LogoLinkProps> = ({ 
  href = '/', 
  onClick,
  ...logoProps 
}) => {
  const handleClick = (e: React.MouseEvent) => {
    if (onClick) {
      e.preventDefault();
      onClick();
    }
  };

  return (
    <a 
      href={href} 
      onClick={handleClick}
      className="inline-block transition-opacity hover:opacity-80 focus:opacity-80 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 rounded"
    >
      <Logo {...logoProps} />
    </a>
  );
};

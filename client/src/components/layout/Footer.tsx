import React from 'react';
import { Link } from 'react-router-dom';
import { Logo } from '@/components/ui/logo';
import { SelaLogo } from '@/components/SelaLogo';
import { Separator } from '@/components/ui/separator';

const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();

  const footerSections = [

    {
      title: 'Sela',
      links: [
        { name: 'About Us', href: 'https://selacloud.com/about' },
        { name: 'Contact', href: 'https://selacloud.com/contact' },
        { name: 'Careers', href: 'https://selacloud.com/careers' },
        { name: 'In The News', href: 'https://selacloud.com/in-the-news' },
      ],
    },
    {
      title: 'Connect',
      links: [
        { name: 'Sela Website', href: 'https://selacloud.com' },
        { name: 'Sela LinkedIn', href: 'https://www.linkedin.com/company/sela-group' },
        { name: 'Ilia German LinkedIn', href: 'https://www.linkedin.com/in/iliagerman/' },
      ],
    },
  ];

  return (
    <footer className="bg-muted/50 border-t">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {/* Logo and description */}
          <div className="lg:col-span-1">
            <Logo className="mb-4" width={150} height={45} />
            <p className="text-sm text-muted-foreground mb-4">
              Your trusted source for breaking news, technology insights, economic analysis, and entertainment coverage.
            </p>
            <p className="text-xs text-muted-foreground">
              © {currentYear} Bob Times. All rights reserved.
            </p>
          </div>

          {/* Footer sections */}
          {footerSections.map((section) => (
            <div key={section.title} className="lg:col-span-1">
              <h3 className="font-semibold text-sm mb-4">{section.title}</h3>
              <ul className="space-y-2">
                {section.links.map((link) => (
                  <li key={link.name}>
                    {link.href.startsWith('http') ? (
                      <a
                        href={link.href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-sm text-muted-foreground hover:text-primary transition-colors"
                      >
                        {link.name}
                      </a>
                    ) : (
                      <Link
                        to={link.href}
                        className="text-sm text-muted-foreground hover:text-primary transition-colors"
                      >
                        {link.name}
                      </Link>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <Separator className="my-8" />

        {/* Bottom section */}
        <div className="flex flex-col sm:flex-row justify-between items-center text-xs text-muted-foreground">
          <div className="mb-4 sm:mb-0">
            <p>
              Made with ❤️ for informed readers worldwide
            </p>
          </div>
          
          {/* Sela Logo in footer */}
          <div className="flex items-center">
            <SelaLogo />
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;

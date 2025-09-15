import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from '@/contexts/ThemeContext';
import { NewspaperProvider } from '@/contexts/NewspaperContext';
import Layout from '@/components/layout/Layout';
import HomePage from '@/components/pages/HomePage';
import TechPage from '@/components/pages/TechPage';
import EconomicsPage from '@/components/pages/EconomicsPage';
import SciencePage from '@/components/pages/SciencePage';
import ArticlePage from '@/components/pages/ArticlePage';
import NewspaperTest from '@/components/NewspaperTest';

function App() {
  return (
    <ThemeProvider>
      <NewspaperProvider>
        <Router>
          <Layout>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/technology" element={<TechPage />} />
              <Route path="/economics" element={<EconomicsPage />} />
              <Route path="/science" element={<SciencePage />} />
              <Route path="/article/:slug" element={<ArticlePage />} />
              <Route path="/newspaper-test" element={<NewspaperTest />} />
            </Routes>
          </Layout>
        </Router>
      </NewspaperProvider>
    </ThemeProvider>
  );
}

export default App;
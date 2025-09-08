# Logo and Icon Assets

This directory contains placeholder logo and icon assets for your application. **All files marked as placeholders should be replaced with your actual brand assets.**

## Current Assets

### Logos
- `logo.svg` - Main application logo (200x60px) - **PLACEHOLDER**
- `logo-light.svg` - Light mode version of logo - **PLACEHOLDER**
- `logo-dark.svg` - Dark mode version of logo - **PLACEHOLDER**

### Icons
- `icon.svg` - Standalone app icon (64x64px) - **PLACEHOLDER**
- `favicon.ico` - Browser tab icon (32x32px) - **PLACEHOLDER** (currently text file)

### PNG Icons (To Be Created)
- `icon-192.png.placeholder` - PWA icon 192x192px - **NEEDS REPLACEMENT**
- `icon-512.png.placeholder` - PWA icon 512x512px - **NEEDS REPLACEMENT**
- `apple-touch-icon.png.placeholder` - iOS icon 180x180px - **NEEDS REPLACEMENT**

### Configuration
- `manifest.json` - PWA manifest file (update with your app details)

## How to Replace Placeholders

### 1. Design Your Brand Assets
Create your logo and icon designs following these guidelines:
- **Logo**: Horizontal layout, readable at small sizes
- **Icon**: Square format, simple design that works at small sizes
- **Colors**: Consider both light and dark mode versions

### 2. Export in Required Formats

#### SVG Files (Vector)
- Export as SVG for scalability
- Keep file sizes reasonable
- Use web-safe fonts or convert text to paths

#### PNG Files (Raster)
- `icon-192.png`: 192x192 pixels
- `icon-512.png`: 512x512 pixels  
- `apple-touch-icon.png`: 180x180 pixels
- Use transparent backgrounds where appropriate
- For Apple touch icon, use solid background (iOS adds effects)

#### ICO File (Favicon)
- `favicon.ico`: 32x32 pixels (can include multiple sizes: 16x16, 32x32, 48x48)
- Use simple, recognizable design at small sizes

### 3. File Replacement Process

1. **Replace SVG files**: Simply overwrite the existing `.svg` files with your designs
2. **Create PNG files**: 
   - Create your PNG files with the exact names (remove `.placeholder` extension)
   - Delete the `.placeholder` files after creating real PNGs
3. **Create favicon.ico**: Replace the text file with a binary ICO file
4. **Update manifest.json**: Modify app name, description, and colors to match your brand

### 4. Recommended Tools

#### Online Tools
- [Favicon.io](https://favicon.io/) - Generate favicons from text, image, or emoji
- [RealFaviconGenerator](https://realfavicongenerator.net/) - Comprehensive favicon generator
- [Convertio](https://convertio.co/svg-png/) - SVG to PNG converter

#### Design Software
- **Figma** (free) - Web-based design tool
- **Adobe Illustrator** - Professional vector graphics
- **Sketch** (Mac) - UI/UX design tool
- **Canva** - Simple online design tool

#### Command Line Tools
```bash
# Convert SVG to PNG using Inkscape
inkscape icon.svg -w 192 -h 192 -o icon-192.png
inkscape icon.svg -w 512 -h 512 -o icon-512.png
inkscape icon.svg -w 180 -h 180 -o apple-touch-icon.png
```

### 5. Testing Your Assets

After replacing the placeholders:
1. **Browser tab**: Check that favicon appears in browser tabs
2. **PWA installation**: Test app installation on mobile devices
3. **iOS bookmark**: Test adding to iOS home screen
4. **Different themes**: Verify logos work in both light and dark modes

### 6. Brand Guidelines

Consider these factors when creating your assets:
- **Consistency**: Use the same colors, fonts, and style across all assets
- **Scalability**: Ensure logos are readable at different sizes
- **Accessibility**: Provide sufficient contrast for readability
- **File sizes**: Keep assets optimized for web performance

## Integration Notes

These assets are automatically referenced in:
- `index.html` - Favicon and icon links
- `manifest.json` - PWA configuration
- React components can import logos using: `import logo from '/logo.svg'`

## Need Help?

If you need assistance with creating or optimizing your brand assets:
1. Consider hiring a designer for professional results
2. Use the recommended online tools for quick solutions
3. Follow web accessibility guidelines for inclusive design
4. Test across different devices and browsers

---

**Remember**: All placeholder files should be replaced before deploying to production!

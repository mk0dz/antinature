import { NextResponse } from 'next/server';
import { getAllContent } from '@/lib/contentUtils';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const query = searchParams.get('q');
  
  if (!query) {
    return NextResponse.json({ error: 'Query parameter is required' }, { status: 400 });
  }
  
  try {
    // Simple search implementation since searchContentFiles isn't available
    const allContent = await getAllContent();
    const searchResults = allContent.filter(content => {
      const title = 'frontmatter' in content ? content.frontmatter.title : content.title;
      const contentText = 'content' in content ? content.content : 
                         ('cells' in content ? content.cells
                           .filter(cell => cell.cell_type === 'markdown')
                           .map(cell => typeof cell.source === 'string' ? cell.source : cell.source.join(''))
                           .join(' ') : '');
      
      const searchTerm = query.toLowerCase();
      return (
        (title && title.toLowerCase().includes(searchTerm)) ||
        (contentText && contentText.toLowerCase().includes(searchTerm))
      );
    });
    
    // Format results
    const formattedResults = searchResults.map(item => {
      const isMarkdown = 'frontmatter' in item;
      const slug = item.slug;
      
      // Properly format the href path based on the content type
      let href = `/docs/${slug}`;
      
      // If the slug already contains a category like 'tutorials/' or 'examples/', 
      // make sure we don't duplicate the path structure
      if (slug.startsWith('tutorials/') || slug.startsWith('examples/')) {
        href = `/docs/${slug}`;
      } else if (slug.includes('/')) {
        // The slug contains a path separator but doesn't start with a category name
        // This might be a nested document, so format appropriately
        href = `/docs/${slug}`;
      }
      
      return {
        title: isMarkdown ? item.frontmatter.title : item.title,
        slug: slug,
        href: href,
        type: isMarkdown ? 'markdown' : 'jupyter',
        excerpt: isMarkdown ? item.content.substring(0, 150) + '...' : 'Jupyter notebook'
      };
    });
    
    return NextResponse.json(formattedResults);
  } catch (error: unknown) {
    console.error('Search error:', error);
    return NextResponse.json({ error: 'Search failed' }, { status: 500 });
  }
} 
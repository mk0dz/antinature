import { getContentBySlug } from '@/lib/contentUtils';
import { isMarkdownContent, isJupyterContent } from '@/lib/contentUtils';
import SimplifiedMDXContent from '@/components/SimplifiedMDXContent';
import { JupyterContentView } from '@/components/JupyterContent';
import { notFound } from 'next/navigation';
import fs from 'fs';
import path from 'path';

// Remove custom interface and use standard Next.js typing

// Helper function to directly read Jupyter file
async function getJupyterFileByPath(filePath: string) {
  try {
    if (fs.existsSync(filePath)) {
      console.log(`Reading Jupyter file directly from: ${filePath}`);
      const fileContent = fs.readFileSync(filePath, 'utf8');
      const notebook = JSON.parse(fileContent);
      
      // Get title from notebook or filename
      let title = path.basename(filePath, '.ipynb');
      for (const cell of notebook.cells) {
        if (cell.cell_type === 'markdown') {
          const source = Array.isArray(cell.source) ? cell.source.join('') : cell.source;
          const match = source.match(/^#\s+(.*)$/m);
          if (match) {
            title = match[1];
            break;
          }
        }
      }
      
      return {
        slug: filePath, // Use path as slug
        title: title,
        cells: notebook.cells
      };
    }
    return null;
  } catch (error) {
    console.error(`Error reading Jupyter file: ${error}`);
    return null;
  }
}

export default async function DocsSlugPage({ params }: { params: { slug: string | string[] } }) {
  // Handle slug differently based on Next.js routing
  const slugPath = typeof params.slug === 'string' ? params.slug : params.slug.join('/');
  
  console.log(`Handling request for route: ${slugPath}`);
  
  // Check if this is a tutorials or examples route
  if (slugPath.startsWith('tutorials/') || slugPath.startsWith('examples/')) {
    const [category, fileName] = slugPath.split('/');
    
    // Special handling for problematic Jupyter files
    const isProblematicJupyter = 
      (category === 'tutorials' && (fileName?.includes('intro_to_antimatter') || fileName?.includes('01_intro')));
    
    if (isProblematicJupyter) {
      console.log(`Using simplified renderer for problematic Jupyter file: ${slugPath}`);
      try {
        // Try to load the content directly
        const contentDir = path.join(process.cwd(), 'src/Content');
        const categoryDir = path.join(contentDir, category);
        
        if (fs.existsSync(categoryDir)) {
          const files = fs.readdirSync(categoryDir)
            .filter(file => file.endsWith('.ipynb') && 
                   (file.includes('intro_to_antimatter') || file.includes('01_intro')));
          
          if (files.length > 0) {
            const filePath = path.join(categoryDir, files[0]);
            const fileContent = fs.readFileSync(filePath, 'utf8');
            
            // Parse as string instead of using the JupyterContentView
            return (
              <div className="prose prose-black max-w-none">
                <h1 className="mb-8 text-4xl font-bold">Introduction to Antimatter</h1>
                <SimplifiedMDXContent content={fileContent} />
              </div>
            );
          }
        }
      } catch (error) {
        console.error('Error loading problematic Jupyter file directly:', error);
      }
    }
    
    // Special direct handling for tutorials and examples without relying on complex resolution
    const contentDir = path.join(process.cwd(), 'src/Content');
    const categoryDir = path.join(contentDir, category);
    
    if (fs.existsSync(categoryDir)) {
      // List all .ipynb files in the category directory
      const files = fs.readdirSync(categoryDir)
        .filter(file => file.endsWith('.ipynb'));
      
      console.log(`Files in ${category} directory:`, files);
      
      // Try different matching strategies:
      
      // 1. Exact match
      const exactMatch = files.find(file => file === `${fileName}.ipynb`);
      if (exactMatch) {
        console.log(`Found exact match: ${exactMatch}`);
        const content = await getJupyterFileByPath(path.join(categoryDir, exactMatch));
        if (content) {
          return (
            <div className="prose prose-black max-w-none">
              <JupyterContentView content={content} />
            </div>
          );
        }
      }
      
      // 2. Match with numeric prefix
      if (fileName.match(/^\d+_/)) {
        // Already has numeric prefix, so we just need to make sure extension is correct
        const prefixMatch = files.find(file => 
          file.toLowerCase() === `${fileName.toLowerCase()}.ipynb`);
        
        if (prefixMatch) {
          console.log(`Found prefix match: ${prefixMatch}`);
          const content = await getJupyterFileByPath(path.join(categoryDir, prefixMatch));
          if (content) {
            return (
              <div className="prose prose-black max-w-none">
                <JupyterContentView content={content} />
              </div>
            );
          }
        }
      }
      
      // 3. Find any file that starts with the same numeric prefix
      const prefixNumMatch = fileName.match(/^(\d+)_/);
      if (prefixNumMatch) {
        const prefix = prefixNumMatch[1];
        const prefixOnlyMatch = files.find(file => file.startsWith(`${prefix}_`));
        
        if (prefixOnlyMatch) {
          console.log(`Found numeric prefix match: ${prefixOnlyMatch}`);
          const content = await getJupyterFileByPath(path.join(categoryDir, prefixOnlyMatch));
          if (content) {
            return (
              <div className="prose prose-black max-w-none">
                <JupyterContentView content={content} />
              </div>
            );
          }
        }
      }
      
      // 4. Partial content match (search for filename without prefix in all files)
      const cleanName = fileName.replace(/^\d+_/, '');
      const partialMatches = files.filter(file => {
        const fileCleanName = file.replace(/^\d+_/, '').replace(/\.ipynb$/, '');
        return fileCleanName.toLowerCase().includes(cleanName.toLowerCase()) ||
               cleanName.toLowerCase().includes(fileCleanName.toLowerCase());
      });
      
      if (partialMatches.length > 0) {
        console.log(`Found partial matches:`, partialMatches);
        
        // Use the first match
        const firstMatch = partialMatches[0];
        const content = await getJupyterFileByPath(path.join(categoryDir, firstMatch));
        
        if (content) {
          return (
            <div className="prose prose-black max-w-none">
              <JupyterContentView content={content} />
            </div>
          );
        }
      }
      
      // If we've tried everything and still not found a match, show file list
      console.error(`No matching file found for ${slugPath} after all attempts`);
      return (
        <div className="prose prose-black max-w-none">
          <h1>File not found</h1>
          <p>Could not find a file matching: {fileName}</p>
          <p>Available files in {category}:</p>
          <ul>
            {files.map(file => (
              <li key={file}>
                {file.replace(/\.ipynb$/, '')}
              </li>
            ))}
          </ul>
        </div>
      );
    }
  }
  
  // Standard content handling for other files
  try {
    const content = await getContentBySlug(slugPath);
    
    if (!content) {
      console.error(`Content not found for slug: ${slugPath}`);
      return notFound();
    }
    
    console.log(`Content found for ${slugPath}, type: ${isMarkdownContent(content) ? 'markdown' : 'jupyter'}`);
    
    // Previously we were only using SimplifiedMDXContent for problematic files,
    // but now we'll use it for ALL markdown content to avoid the MDX processing errors
    return (
      <div className="prose prose-black max-w-none">
        {isMarkdownContent(content) && (
          <>
            <h1 className="mb-8 text-4xl font-bold">{content.frontmatter.title || "Documentation"}</h1>
            <SimplifiedMDXContent content={content.content} />
          </>
        )}
        {isJupyterContent(content) && <JupyterContentView content={content} />}
        {!isMarkdownContent(content) && !isJupyterContent(content) && (
          <div className="p-4 text-red-500 border border-red-300 rounded">
            <h2>Error: Unsupported content format</h2>
            <p>The content format is neither Markdown nor Jupyter notebook.</p>
          </div>
        )}
      </div>
    );
  } catch (error) {
    console.error(`Error loading content for slug: ${slugPath}`, error);
    return (
      <div className="p-4 text-red-500 border border-red-300 rounded">
        <h2>Error loading content</h2>
        <p>There was an error loading the requested content. Please try again later.</p>
      </div>
    );
  }
} 
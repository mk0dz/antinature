import fs from 'fs';
import path from 'path';
import { JupyterContentView } from '@/components/JupyterContent';
import { notFound } from 'next/navigation';

interface ExamplePageParams {
  params: {
    filename: string[];
  };
}

// Function to read Jupyter file directly
async function getJupyterFile(filePath: string) {
  try {
    if (fs.existsSync(filePath)) {
      console.log(`Reading Jupyter file from: ${filePath}`);
      const fileContent = fs.readFileSync(filePath, 'utf8');
      const notebook = JSON.parse(fileContent);
      
      // Extract title from notebook if available
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
        slug: filePath,
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

export default async function ExamplePage({ params }: ExamplePageParams) {
  const filenameArray = params.filename || [];
  const filename = filenameArray.join('/');
  
  console.log(`Examples - looking for file: ${filename}`);
  
  // Locate the file
  const contentDir = path.join(process.cwd(), 'src/Content');
  const examplesDir = path.join(contentDir, 'examples');
  
  if (!fs.existsSync(examplesDir)) {
    console.error('Examples directory does not exist');
    return notFound();
  }
  
  // Get all example files for matching
  const exampleFiles = fs.readdirSync(examplesDir)
    .filter(file => file.endsWith('.ipynb'));
  
  console.log(`Found ${exampleFiles.length} example files:`, exampleFiles);
  
  // Try to find exact match first
  let matchingFile: string | null = null;
  
  // Exact match including .ipynb extension
  if (exampleFiles.includes(`${filename}.ipynb`)) {
    matchingFile = path.join(examplesDir, `${filename}.ipynb`);
  } 
  // Match without extension
  else if (exampleFiles.some(file => file === filename)) {
    matchingFile = path.join(examplesDir, filename);
  }
  // Try to match with numeric prefix
  else {
    // If filename already has a numeric prefix
    if (filename.match(/^\d+_/)) {
      const exactMatch = exampleFiles.find(file => 
        file.toLowerCase() === `${filename.toLowerCase()}.ipynb` || 
        file.toLowerCase() === filename.toLowerCase());
      
      if (exactMatch) {
        matchingFile = path.join(examplesDir, exactMatch);
      }
    } 
    // Try to find file that starts with same numeric prefix
    else if (filename.match(/^\d+$/)) {
      const prefixMatch = exampleFiles.find(file => file.startsWith(`${filename}_`));
      if (prefixMatch) {
        matchingFile = path.join(examplesDir, prefixMatch);
      }
    }
    // Look for files that match the name without any prefix
    else {
      const partialMatches = exampleFiles.filter(file => {
        const fileWithoutPrefix = file.replace(/^\d+_/, '').replace(/\.ipynb$/, '');
        return fileWithoutPrefix.toLowerCase() === filename.toLowerCase() ||
               file.toLowerCase().includes(filename.toLowerCase());
      });
      
      if (partialMatches.length > 0) {
        matchingFile = path.join(examplesDir, partialMatches[0]);
      }
    }
  }
  
  if (!matchingFile || !fs.existsSync(matchingFile)) {
    console.error(`No matching example file found for: ${filename}`);
    return (
      <div className="prose prose-black max-w-none">
        <h1>Example Not Found</h1>
        <p>Could not find an example matching: {filename}</p>
        <h2>Available Examples:</h2>
        <ul>
          {exampleFiles.map(file => (
            <li key={file}>
              {file.replace(/\.ipynb$/, '')}
            </li>
          ))}
        </ul>
      </div>
    );
  }
  
  // Found a matching file, read it
  const content = await getJupyterFile(matchingFile);
  
  if (!content) {
    console.error(`Error reading example file: ${matchingFile}`);
    return notFound();
  }
  
  return (
    <div className="prose prose-black max-w-none">
      <JupyterContentView content={content} />
    </div>
  );
} 
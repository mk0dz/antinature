import fs from 'fs';
import path from 'path';
import { JupyterContentView } from '@/components/JupyterContent';
import { notFound } from 'next/navigation';

interface TutorialPageParams {
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

export default async function TutorialPage({ params }: TutorialPageParams) {
  const filenameArray = params.filename || [];
  const filename = filenameArray.join('/');
  
  console.log(`Tutorials - looking for file: ${filename}`);
  
  // Locate the file
  const contentDir = path.join(process.cwd(), 'src/Content');
  const tutorialsDir = path.join(contentDir, 'tutorials');
  
  if (!fs.existsSync(tutorialsDir)) {
    console.error('Tutorials directory does not exist');
    return notFound();
  }
  
  // Get all tutorial files for matching
  const tutorialFiles = fs.readdirSync(tutorialsDir)
    .filter(file => file.endsWith('.ipynb'));
  
  console.log(`Found ${tutorialFiles.length} tutorial files:`, tutorialFiles);
  
  // Try to find exact match first
  let matchingFile: string | null = null;
  
  // Exact match including .ipynb extension
  if (tutorialFiles.includes(`${filename}.ipynb`)) {
    matchingFile = path.join(tutorialsDir, `${filename}.ipynb`);
  } 
  // Match without extension
  else if (tutorialFiles.some(file => file === filename)) {
    matchingFile = path.join(tutorialsDir, filename);
  }
  // Try to match with numeric prefix
  else {
    // If filename already has a numeric prefix
    if (filename.match(/^\d+_/)) {
      const exactMatch = tutorialFiles.find(file => 
        file.toLowerCase() === `${filename.toLowerCase()}.ipynb` || 
        file.toLowerCase() === filename.toLowerCase());
      
      if (exactMatch) {
        matchingFile = path.join(tutorialsDir, exactMatch);
      }
    } 
    // Try to find file that starts with same numeric prefix
    else if (filename.match(/^\d+$/)) {
      const prefixMatch = tutorialFiles.find(file => file.startsWith(`${filename}_`));
      if (prefixMatch) {
        matchingFile = path.join(tutorialsDir, prefixMatch);
      }
    }
    // Look for files that match the name without any prefix
    else {
      const partialMatches = tutorialFiles.filter(file => {
        const fileWithoutPrefix = file.replace(/^\d+_/, '').replace(/\.ipynb$/, '');
        return fileWithoutPrefix.toLowerCase() === filename.toLowerCase() ||
               file.toLowerCase().includes(filename.toLowerCase());
      });
      
      if (partialMatches.length > 0) {
        matchingFile = path.join(tutorialsDir, partialMatches[0]);
      }
    }
  }
  
  if (!matchingFile || !fs.existsSync(matchingFile)) {
    console.error(`No matching tutorial file found for: ${filename}`);
    return (
      <div className="prose prose-black max-w-none">
        <h1>Tutorial Not Found</h1>
        <p>Could not find a tutorial matching: {filename}</p>
        <h2>Available Tutorials:</h2>
        <ul>
          {tutorialFiles.map(file => (
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
    console.error(`Error reading tutorial file: ${matchingFile}`);
    return notFound();
  }
  
  return (
    <div className="prose prose-black max-w-none">
      <JupyterContentView content={content} />
    </div>
  );
} 
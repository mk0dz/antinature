import fs from 'fs';
import path from 'path';
import Link from 'next/link';

export default function DebugPage() {
  // Direct file system reading for debugging
  const contentDir = path.join(process.cwd(), 'src/Content');
  
  let tutorials: string[] = [];
  let examples: string[] = [];
  
  // Read tutorials directory
  const tutorialsDir = path.join(contentDir, 'tutorials');
  if (fs.existsSync(tutorialsDir)) {
    tutorials = fs.readdirSync(tutorialsDir)
      .filter(file => file.endsWith('.ipynb'));
  }
  
  // Read examples directory
  const examplesDir = path.join(contentDir, 'examples');
  if (fs.existsSync(examplesDir)) {
    examples = fs.readdirSync(examplesDir)
      .filter(file => file.endsWith('.ipynb'));
  }
  
  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold mb-6">Debug Page - Files in Content Directory</h1>
      
      <p className="mb-4">Content directory: {contentDir}</p>
      
      <div className="mb-8">
        <h2 className="text-xl font-semibold mb-4">Tutorials ({tutorials.length} files):</h2>
        <div className="grid grid-cols-2 gap-4">
          {tutorials.map(file => {
            const fileName = file.replace(/\.ipynb$/, '');
            const filePath = path.join(tutorialsDir, file);
            const fileStats = fs.statSync(filePath);
            
            // Create link directly to this file in multiple formats
            return (
              <div key={file} className="border p-4 rounded">
                <h3 className="font-medium">{fileName}</h3>
                <p className="text-sm mb-2">Size: {fileStats.size} bytes</p>
                <p className="text-sm mb-3">Path: {filePath}</p>
                <div className="space-y-2">
                  <p className="text-sm font-semibold">Try these links:</p>
                  <Link href={`/docs/tutorials/${fileName}`} className="block text-blue-500 hover:underline text-sm">
                    /docs/tutorials/{fileName}
                  </Link>
                  {fileName.match(/^\d+_/) && (
                    <Link href={`/docs/tutorials/${fileName.replace(/^\d+_/, '')}`} className="block text-blue-500 hover:underline text-sm">
                      /docs/tutorials/{fileName.replace(/^\d+_/, '')}
                    </Link>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
      
      <div className="mb-8">
        <h2 className="text-xl font-semibold mb-4">Examples ({examples.length} files):</h2>
        <div className="grid grid-cols-2 gap-4">
          {examples.map(file => {
            const fileName = file.replace(/\.ipynb$/, '');
            const filePath = path.join(examplesDir, file);
            const fileStats = fs.statSync(filePath);
            
            // Create link directly to this file in multiple formats
            return (
              <div key={file} className="border p-4 rounded">
                <h3 className="font-medium">{fileName}</h3>
                <p className="text-sm mb-2">Size: {fileStats.size} bytes</p>
                <p className="text-sm mb-3">Path: {filePath}</p>
                <div className="space-y-2">
                  <p className="text-sm font-semibold">Try these links:</p>
                  <Link href={`/docs/examples/${fileName}`} className="block text-blue-500 hover:underline text-sm">
                    /docs/examples/{fileName}
                  </Link>
                  {fileName.match(/^\d+_/) && (
                    <Link href={`/docs/examples/${fileName.replace(/^\d+_/, '')}`} className="block text-blue-500 hover:underline text-sm">
                      /docs/examples/{fileName.replace(/^\d+_/, '')}
                    </Link>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
} 
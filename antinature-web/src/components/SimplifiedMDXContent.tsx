'use client';

import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vs } from 'react-syntax-highlighter/dist/esm/styles/prism';
import remarkGfm from 'remark-gfm';
import 'katex/dist/katex.min.css';
import katex from 'katex';
import mermaid from 'mermaid';

// Initialize mermaid with default settings
mermaid.initialize({
  startOnLoad: true,
  theme: 'default',
  securityLevel: 'loose',
  flowchart: {
    htmlLabels: true,
    curve: 'linear',
  },
  fontSize: 14, // Smaller font size by default
});

interface SimplifiedMDXContentProps {
  content: string;
}

// Define a custom Text component that will look for math placeholders
const MathText = ({ 
  text, 
  mathExpressions 
}: { 
  text: string, 
  mathExpressions: {[key: string]: string} 
}) => {
  if (!text) return null;

  // Try to match any math placeholders in the text
  const displayMathRegex = /DISPLAY_MATH_\d+/g;
  const inlineMathRegex = /INLINE_MATH_\d+/g;
  
  // Check if this text node contains any math placeholders
  const containsDisplayMath = displayMathRegex.test(text);
  const containsInlineMath = inlineMathRegex.test(text);
  
  // If there's no math, just return the text as is
  if (!containsDisplayMath && !containsInlineMath) {
    return <>{text}</>;
  }
  
  // For a simple case where the text is just a placeholder
  if (text.match(/^(DISPLAY_MATH_\d+|INLINE_MATH_\d+)$/)) {
    const isDisplay = text.startsWith('DISPLAY_MATH_');
    const formula = mathExpressions[text];
    
    if (!formula) {
      console.error(`Math expression not found for placeholder: ${text}`);
      return <span className="math-error text-red-600">Missing math: {text}</span>;
    }
    
    try {
      const html = katex.renderToString(formula, {
        displayMode: isDisplay,
        throwOnError: false,
        strict: false
      });
      
      if (isDisplay) {
        return (
          <div className="math-display my-4 text-center">
            <span dangerouslySetInnerHTML={{ __html: html }} />
          </div>
        );
      } else {
        return <span className="math-inline" dangerouslySetInnerHTML={{ __html: html }} />;
      }
    } catch (err) {
      console.error(`Error rendering ${isDisplay ? 'display' : 'inline'} math:`, err);
      return (
        <span className="math-error text-red-600">
          Error rendering math: {formula || text}
        </span>
      );
    }
  }
  
  // For more complex cases where the text contains multiple placeholders or mixed content
  let result: React.ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  
  // Combined regex to find both types of placeholders
  const combinedRegex = /(DISPLAY_MATH_\d+|INLINE_MATH_\d+)/g;
  
  while ((match = combinedRegex.exec(text)) !== null) {
    // Add text before the match
    if (match.index > lastIndex) {
      result.push(<span key={`text-${lastIndex}`}>{text.substring(lastIndex, match.index)}</span>);
    }
    
    // Add the math component
    const placeholder = match[0];
    const isDisplay = placeholder.startsWith('DISPLAY_MATH_');
    const formula = mathExpressions[placeholder];
    
    if (!formula) {
      result.push(
        <span key={`placeholder-${match.index}`} className="math-error text-red-600">
          Missing math: {placeholder}
        </span>
      );
    } else {
      try {
        const html = katex.renderToString(formula, {
          displayMode: isDisplay,
          throwOnError: false,
          strict: false
        });
        
        if (isDisplay) {
          result.push(
            <div key={`math-${match.index}`} className="math-display my-4 text-center">
              <span dangerouslySetInnerHTML={{ __html: html }} />
            </div>
          );
        } else {
          result.push(
            <span 
              key={`math-${match.index}`} 
              className="math-inline" 
              dangerouslySetInnerHTML={{ __html: html }} 
            />
          );
        }
      } catch (err) {
        result.push(
          <span key={`error-${match.index}`} className="math-error text-red-600">
            Error rendering math: {formula}
          </span>
        );
      }
    }
    
    lastIndex = match.index + match[0].length;
  }
  
  // Add any remaining text
  if (lastIndex < text.length) {
    result.push(<span key={`text-${lastIndex}`}>{text.substring(lastIndex)}</span>);
  }
  
  return <>{result}</>;
};

// Custom paragraph component to handle math expressions
const CustomParagraph = ({ 
  children, 
  mathExpressions 
}: { 
  children: React.ReactNode, 
  mathExpressions: {[key: string]: string} 
}) => {
  // Process all children
  const processedChildren = React.Children.toArray(children).map((child, index) => {
    if (typeof child === 'string') {
      return <MathText key={index} text={child} mathExpressions={mathExpressions} />;
    }
    return child;
  });
  
  // Use a div instead of p to avoid HTML nesting issues
  return <div className="mb-4 text-base leading-7">{processedChildren}</div>;
};

// Text component to process any text node in markdown
const CustomText = ({ 
  children, 
  mathExpressions 
}: { 
  children: React.ReactNode, 
  mathExpressions: {[key: string]: string} 
}) => {
  if (typeof children === 'string') {
    return <MathText text={children} mathExpressions={mathExpressions} />;
  }
  return <>{children}</>;
};

// Mermaid component to render diagrams
const MermaidDiagram = ({ content }: { content: string }) => {
  const mermaidRef = useRef<HTMLDivElement>(null);
  const [diagram, setDiagram] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [scale, setScale] = useState<number>(0.8); // Start with 80% size
  const [isPanning, setIsPanning] = useState<boolean>(false);
  const [position, setPosition] = useState<{ x: number, y: number }>({ x: 0, y: 0 });
  const [startPanPosition, setStartPanPosition] = useState<{ x: number, y: number }>({ x: 0, y: 0 });

  useEffect(() => {
    if (mermaidRef.current) {
      try {
        // Don't re-initialize here, just render
        mermaid.render(`mermaid-${Date.now()}`, content)
          .then(({ svg }) => {
            setDiagram(svg);
            setError(null);
          })
          .catch((err) => {
            console.error('Mermaid rendering error:', err);
            setError(`Error rendering diagram: ${err.message}`);
          });
      } catch (err: any) {
        console.error('Mermaid rendering error:', err);
        setError(`Error rendering diagram: ${err.message}`);
      }
    }
  }, [content, mermaidRef]);

  const handleZoomIn = () => {
    setScale(prevScale => Math.min(prevScale + 0.1, 2.0)); // Max zoom: 200%
  };

  const handleZoomOut = () => {
    setScale(prevScale => Math.max(prevScale - 0.1, 0.3)); // Min zoom: 30%
  };

  const handleReset = () => {
    setScale(0.8);
    setPosition({ x: 0, y: 0 });
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0) { // Left mouse button
      setIsPanning(true);
      setStartPanPosition({ x: e.clientX - position.x, y: e.clientY - position.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isPanning) {
      setPosition({
        x: e.clientX - startPanPosition.x,
        y: e.clientY - startPanPosition.y
      });
    }
  };

  const handleMouseUp = () => {
    setIsPanning(false);
  };

  const handleMouseLeave = () => {
    setIsPanning(false);
  };

  if (error) {
    return (
      <div className="mermaid-error p-4 bg-red-50 border border-red-200 rounded text-red-600">
        <p className="font-bold">Diagram Error</p>
        <p>{error}</p>
        <pre className="mt-2 p-2 bg-gray-100 overflow-auto text-sm">{content}</pre>
      </div>
    );
  }

  return (
    <div className="mermaid-container my-6 relative">
      <div className="zoom-controls absolute top-0 right-0 bg-white border border-gray-200 rounded p-1 z-10 flex space-x-1">
        <button 
          onClick={handleZoomIn} 
          className="p-1 hover:bg-gray-100 rounded"
          title="Zoom in"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="11" cy="11" r="8"></circle>
            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
            <line x1="11" y1="8" x2="11" y2="14"></line>
            <line x1="8" y1="11" x2="14" y2="11"></line>
          </svg>
        </button>
        <button 
          onClick={handleZoomOut} 
          className="p-1 hover:bg-gray-100 rounded"
          title="Zoom out"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="11" cy="11" r="8"></circle>
            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
            <line x1="8" y1="11" x2="14" y2="11"></line>
          </svg>
        </button>
        <button 
          onClick={handleReset} 
          className="p-1 hover:bg-gray-100 rounded"
          title="Reset view"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path>
            <path d="M3 3v5h5"></path>
          </svg>
        </button>
      </div>
      <div className="mermaid-wrapper overflow-hidden border border-gray-200 rounded" style={{ 
        height: '500px', 
        maxHeight: '80vh',
        position: 'relative',
        backgroundColor: '#fff'
      }}>
        <div 
          className="mermaid-diagram cursor-move absolute"
          style={{ 
            transform: `scale(${scale}) translate(${position.x / scale}px, ${position.y / scale}px)`,
            transformOrigin: 'center center',
            transition: isPanning ? 'none' : 'transform 0.2s ease',
            minWidth: '100%',
            minHeight: '100%',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center'
          }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
          ref={mermaidRef}
          dangerouslySetInnerHTML={{ __html: diagram }}
        />
      </div>
      <div className="text-center text-xs text-gray-500 mt-2">
        Zoom: {Math.round(scale * 100)}% • Click and drag to pan • Use controls to zoom
      </div>
    </div>
  );
};

export default function SimplifiedMDXContent({ content }: SimplifiedMDXContentProps) {
  const [sanitizedContent, setSanitizedContent] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [mathExpressions, setMathExpressions] = useState<{[key: string]: string}>({});
  const [showDebugInfo, setShowDebugInfo] = useState<boolean>(false);

  const insertTestMath = () => {
    // Create a test math expression
    const testMathExps: {[key: string]: string} = {
      'INLINE_MATH_0': '\\alpha',
      'INLINE_MATH_1': 'c',
      'INLINE_MATH_2': '\\delta(x)',
      'DISPLAY_MATH_0': 'E = mc^2',
      'DISPLAY_MATH_1': '\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}'
    };
    
    // Update state with test expressions
    setMathExpressions(testMathExps);
    // Create test content with placeholders
    setSanitizedContent(
      '# Math Test\n\n' +
      'Here are some inline math examples: INLINE_MATH_0 is the fine structure constant, ' +
      'INLINE_MATH_1 is the speed of light, and INLINE_MATH_2 is the Dirac delta function.\n\n' +
      'And here is a display math example:\n\n' +
      'DISPLAY_MATH_0\n\n' +
      'Another beautiful equation:\n\n' +
      'DISPLAY_MATH_1'
    );
    
    // Show debug info automatically
    setShowDebugInfo(true);
  };

  useEffect(() => {
    if (!content) {
      setError('No content provided');
      setIsLoading(false);
      return;
    }

    try {
      // Check if content is a JSON string from a Jupyter notebook
      let processedContent = content;
      
      if (content.trim().startsWith('{') && content.includes('"cells"')) {
        processedContent = extractTextFromJupyterNotebook(content);
      }
      
      // Apply aggressive sanitization to the content
      const { sanitized, mathExps } = sanitizeMarkdownWithMath(processedContent);
      setSanitizedContent(sanitized);
      setMathExpressions(mathExps);
    } catch (err: unknown) {
      console.error('Error sanitizing content:', err);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(`Error preparing content: ${errorMessage}`);
      // Still try to render the original content as a fallback
      setSanitizedContent(content);
    }
    
    setIsLoading(false);
  }, [content]);

  // Helper function to sanitize markdown and extract math expressions
  function sanitizeMarkdownWithMath(markdown: string): { sanitized: string, mathExps: {[key: string]: string} } {
    console.log('Sanitizing markdown content with math extraction...');
    let sanitized = markdown;
    const mathExps: {[key: string]: string} = {};
    
    // Replace code blocks first to protect them from processing
    const codeBlocks: string[] = [];
    sanitized = sanitized.replace(/```[\s\S]*?```/g, (match) => {
      const placeholder = `CODE_BLOCK_${codeBlocks.length}`;
      codeBlocks.push(match);
      return placeholder;
    });
    
    // Extract display math ($$...$$)
    let displayMathCounter = 0;
    sanitized = sanitized.replace(/\$\$([\s\S]*?)\$\$/g, (match, expr) => {
      const placeholder = `DISPLAY_MATH_${displayMathCounter}`;
      mathExps[placeholder] = expr.trim();
      displayMathCounter++;
      return placeholder; // Return just the placeholder without extra characters
    });
    
    // Extract inline math ($...$)
    let inlineMathCounter = 0;
    sanitized = sanitized.replace(/\$([^\$\n]+?)\$/g, (match, expr) => {
      const placeholder = `INLINE_MATH_${inlineMathCounter}`;
      mathExps[placeholder] = expr.trim();
      inlineMathCounter++;
      return placeholder; // Return just the placeholder without extra characters
    });
    
    // Fix problematic table cells like empty cells
    sanitized = sanitized.replace(/\|\s*\|/g, '| |');
    
    // Aggressively handle tables
    // First, add empty lines around tables
    sanitized = sanitized.replace(/([^\n])\n(\|(?:.*\|)+)/g, '$1\n\n$2');
    sanitized = sanitized.replace(/(\|(?:.*\|)+)\n([^\n])/g, '$1\n\n$2');
    
    // Fix table rows with inconsistent cell counts
    const tableRegex = /(\|(?:.*\|)+\n)+/g;
    sanitized = sanitized.replace(tableRegex, (tableBlock) => {
      try {
        // Split into rows and process
        const rows = tableBlock.trim().split('\n');
        if (rows.length < 2) return tableBlock; // Not a real table
        
        // Count cells in each row
        const cellCounts = rows.map(row => (row.match(/\|/g) || []).length - 1);
        const maxCells = Math.max(...cellCounts);
        
        // Fix rows with fewer cells
        const fixedRows = rows.map((row) => {
          const currentCells = (row.match(/\|/g) || []).length - 1;
          if (currentCells < maxCells) {
            return row.replace(/\|$/, ' |'.repeat(maxCells - currentCells) + '|');
          }
          return row;
        });
        
        // Ensure there's a separator row if this looks like a table with a header
        if (rows.length >= 2 && !rows[1].includes('-')) {
          const separatorRow = '|' + ' --- |'.repeat(maxCells);
          fixedRows.splice(1, 0, separatorRow);
        }
        
        return fixedRows.join('\n') + '\n';
      } catch (err) {
        console.error('Error processing table:', err);
        // If anything goes wrong, return a simplified version of the table
        return tableBlock.replace(/\|/g, '\\|');
      }
    });
    
    // Process headings to ensure proper spacing
    sanitized = sanitized.replace(/^(#{1,6})\s*(.+)$/gm, (match, hashes, text) => {
      return `${hashes} ${text.trim()}`;
    });
    
    // Ensure there's a blank line before headings
    sanitized = sanitized.replace(/([^\n])\n(#{1,6}\s)/g, '$1\n\n$2');
    
    // Restore code blocks
    sanitized = sanitized.replace(/CODE_BLOCK_(\d+)/g, (_, index) => 
      codeBlocks[parseInt(index)] || '```\nCode block error\n```'
    );
    
    // Remove problematic imports/exports
    sanitized = sanitized.replace(/import\s+.*\s+from\s+["'].*["']/g, '<!-- import removed -->');
    sanitized = sanitized.replace(/export\s+.*{.*}/g, '<!-- export removed -->');
    
    // Remove problematic HTML that react-markdown might not handle well
    sanitized = sanitized.replace(/<(script|style|iframe)[\s\S]*?<\/\1>/gi, '<!-- removed -->');
    
    // Ensure proper spacing for list items
    sanitized = sanitized.replace(/^(\s*)([*+-]|\d+\.)\s*/gm, '$1$2 ');
    
    console.log('Sanitization complete with math extraction');
    console.log(`Extracted ${displayMathCounter} display math and ${inlineMathCounter} inline math expressions`);
    console.log('Math expressions:', mathExps);
    return { sanitized, mathExps };
  }

  // Function to extract markdown and code text from Jupyter notebook JSON
  function extractTextFromJupyterNotebook(notebookJson: string): string {
    try {
      console.log('Parsing Jupyter notebook JSON');
      const notebook = JSON.parse(notebookJson) as {
        cells: Array<{ 
          cell_type: string; 
          source: string | string[]; 
          outputs?: Array<{
            output_type: string;
            text?: string | string[];
            data?: Record<string, string | string[]>;
            traceback?: string[];
            ename?: string;
            evalue?: string;
          }>;
        }>;
        metadata?: { 
          language_info?: { 
            name: string 
          } 
        };
      };
      let extractedContent = '';
      
      if (Array.isArray(notebook.cells)) {
        console.log(`Processing ${notebook.cells.length} cells from notebook`);
        notebook.cells.forEach((cell, index) => {
          try {
            if (cell.cell_type === 'markdown') {
              // Extract markdown content
              const source = Array.isArray(cell.source) ? cell.source.join('') : cell.source;
              extractedContent += source + '\n\n';
            } else if (cell.cell_type === 'code') {
              // For code cells, wrap in code blocks
              const source = Array.isArray(cell.source) ? cell.source.join('') : cell.source;
              if (source.trim()) {
                const language = notebook.metadata?.language_info?.name || 'python';
                extractedContent += '```' + language + '\n' + source + '\n```\n\n';
              }
              
              // If there are outputs, extract text and images
              if (Array.isArray(cell.outputs) && cell.outputs.length > 0) {
                cell.outputs.forEach((output: any) => {
                  if (output.output_type === 'stream' && output.text) {
                    const text = Array.isArray(output.text) ? output.text.join('') : output.text;
                    extractedContent += '```\n' + text + '\n```\n\n';
                  } else if (output.output_type === 'execute_result' && output.data) {
                    // Handle text/plain output
                    if (output.data['text/plain']) {
                      const text = Array.isArray(output.data['text/plain']) 
                        ? output.data['text/plain'].join('') 
                        : output.data['text/plain'];
                      extractedContent += '```\n' + text + '\n```\n\n';
                    }
                    
                    // Handle text/html output (convert to markdown)
                    if (output.data['text/html']) {
                      extractedContent += '<div class="output-html">\n';
                      extractedContent += Array.isArray(output.data['text/html']) 
                        ? output.data['text/html'].join('') 
                        : output.data['text/html'];
                      extractedContent += '\n</div>\n\n';
                    }
                  } else if (output.output_type === 'display_data' && output.data) {
                    // Similar to execute_result
                    if (output.data['text/plain']) {
                      const text = Array.isArray(output.data['text/plain']) 
                        ? output.data['text/plain'].join('') 
                        : output.data['text/plain'];
                      extractedContent += '```\n' + text + '\n```\n\n';
                    }
                    
                    if (output.data['text/html']) {
                      extractedContent += '<div class="output-html">\n';
                      extractedContent += Array.isArray(output.data['text/html']) 
                        ? output.data['text/html'].join('') 
                        : output.data['text/html'];
                      extractedContent += '\n</div>\n\n';
                    }
                    
                    // Handle image output by converting to markdown image syntax
                    if (output.data['image/png']) {
                      const imageData = output.data['image/png'];
                      extractedContent += `![Output ${index}](data:image/png;base64,${imageData})\n\n`;
                    }
                  } else if (output.output_type === 'error') {
                    // Handle error outputs
                    if (Array.isArray(output.traceback)) {
                      extractedContent += '```\n' + output.traceback.join('\n') + '\n```\n\n';
                    } else if (output.ename && output.evalue) {
                      extractedContent += `**Error**: ${output.ename}: ${output.evalue}\n\n`;
                    }
                  }
                });
              }
            } else if (cell.cell_type === 'raw') {
              // Just add raw content as-is
              const source = Array.isArray(cell.source) ? cell.source.join('') : cell.source;
              extractedContent += source + '\n\n';
            }
          } catch (cellError) {
            console.error(`Error processing cell ${index}:`, cellError);
            extractedContent += `\n\n**Error processing cell ${index}**\n\n`;
          }
        });
      }
      
      if (!extractedContent.trim()) {
        console.warn('No content was extracted from the notebook');
        return '# Jupyter Notebook\n\nUnable to extract meaningful content from this notebook.';
      }
      
      console.log(`Successfully extracted ${extractedContent.length} characters of markdown content`);
      return extractedContent;
    } catch (err) {
      console.error('Error extracting text from Jupyter notebook:', err);
      return '# Error Processing Jupyter Notebook\n\nThere was an error processing this notebook file.';
    }
  }

  if (isLoading) {
    return <div>Loading content...</div>;
  }

  if (error && !sanitizedContent) {
    return (
      <div className="mdx-content error">
        <h2 className="text-xl font-bold text-red-500 mb-4">Content Rendering Error</h2>
        <p className="mb-4">{error}</p>
      </div>
    );
  }

  return (
    <div className="mdx-content">
      <div className="flex justify-end mb-2 space-x-2">
        <button 
          onClick={insertTestMath}
          className="text-xs bg-blue-100 hover:bg-blue-200 text-blue-700 px-2 py-1 rounded"
        >
          Test Math Rendering
        </button>
        <button 
          onClick={() => setShowDebugInfo(!showDebugInfo)}
          className="text-xs bg-gray-100 hover:bg-gray-200 text-gray-700 px-2 py-1 rounded"
        >
          {showDebugInfo ? 'Hide Debug Info' : 'Show Debug Info'}
        </button>
      </div>
      
      {showDebugInfo && (
        <div className="mb-4 p-3 bg-gray-100 rounded text-xs font-mono overflow-auto max-h-60">
          <h3 className="font-bold mb-1">Math Expressions ({Object.keys(mathExpressions).length}):</h3>
          <div className="mb-2">
            {Object.entries(mathExpressions).map(([key, value], index) => (
              <div key={index} className="mb-1 pb-1 border-b border-gray-200">
                <strong>{key}:</strong> "{value}"
              </div>
            ))}
          </div>
          <h3 className="font-bold mt-2 mb-1">Content Check:</h3>
          <div className="mb-2">
            <div>Display Math Placeholders: {sanitizedContent.match(/DISPLAY_MATH_\d+/g)?.length || 0}</div>
            <div>Inline Math Placeholders: {sanitizedContent.match(/INLINE_MATH_\d+/g)?.length || 0}</div>
          </div>
          <h3 className="font-bold mt-2 mb-1">Sanitized Content Preview:</h3>
          <pre className="whitespace-pre-wrap">{sanitizedContent.substring(0, 300)}...</pre>
        </div>
      )}
      
      <div className="markdown-body">
        {(() => {
          try {
            return (
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  // Heading styles
                  h1: (props) => <h1 className="mb-6 mt-10 text-3xl font-bold text-black" {...props} />,
                  h2: (props) => <h2 className="mb-4 mt-8 text-2xl font-bold text-black" {...props} />,
                  h3: (props) => <h3 className="mb-3 mt-6 text-xl font-bold text-black" {...props} />,
                  h4: (props) => <h4 className="mb-2 mt-4 text-lg font-bold text-black" {...props} />,
                  
                  // Paragraph and inline styles
                  p: ({ children }) => (
                    <CustomParagraph mathExpressions={mathExpressions}>{children}</CustomParagraph>
                  ),
                  a: (props) => <a className="text-blue-600 hover:underline" {...props} />,
                  ul: (props) => <ul className="mb-4 ml-6 list-disc" {...props} />,
                  ol: (props) => <ol className="mb-4 ml-6 list-decimal" {...props} />,
                  li: (props) => <li className="mb-1" {...props} />,
                  
                  // Text component for all text nodes
                  text: ({ children }) => (
                    <CustomText mathExpressions={mathExpressions}>{children}</CustomText>
                  ),
                  
                  // Code components
                  code: ({ inline, className, children, ...props }) => {
                    const match = /language-(\w+)/.exec(className || '');
                    
                    // Handle Mermaid diagrams
                    if (!inline && match && match[1] === 'mermaid') {
                      return <MermaidDiagram content={String(children).replace(/\n$/, '')} />;
                    }
                    
                    return !inline && match ? (
                      <SyntaxHighlighter
                        language={match[1]}
                        style={vs}
                        PreTag="div"
                        className="mb-4 rounded border border-gray-200 text-gray-800"
                        codeTagProps={{
                          style: {
                            fontSize: '0.9rem',
                            fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
                            color: '#333',
                          }
                        }}
                        customStyle={{
                          backgroundColor: 'white',
                          padding: '1rem',
                          color: '#333',
                        }}
                        // Override the vs theme with custom token styling
                        lineProps={{
                          style: { backgroundColor: 'white' }
                        }}
                        {...props}
                      >
                        {String(children).replace(/\n$/, '')}
                      </SyntaxHighlighter>
                    ) : (
                      <code className="rounded bg-white border border-gray-100 px-1 py-0.5 font-mono text-sm text-gray-800" {...props}>
                        {children}
                      </code>
                    );
                  },
                  
                  // Table components
                  table: (props) => (
                    <div className="mb-4 overflow-x-auto">
                      <table className="w-full border-collapse" {...props} />
                    </div>
                  ),
                  th: (props) => (
                    <th className="border border-gray-300 bg-gray-100 px-4 py-2 text-left font-medium" {...props} />
                  ),
                  td: (props) => (
                    <td className="border border-gray-300 px-4 py-2" {...props} />
                  ),
                  
                  // Other components
                  blockquote: (props) => (
                    <blockquote className="mb-4 border-l-4 border-gray-200 pl-4 italic" {...props} />
                  ),
                  img: (props) => (
                    <img className="mb-4 max-w-full" alt={props.alt || ''} {...props} />
                  ),
                  hr: () => <hr className="my-6 border-t border-gray-300" />,
                }}
              >
                {sanitizedContent}
              </ReactMarkdown>
            );
          } catch (renderError) {
            console.error('Error rendering markdown with math:', renderError);
            
            // Super basic fallback renderer
            return (
              <div className="markdown-error">
                <h2 className="text-xl font-bold text-red-500 mb-4">Error Rendering Content</h2>
                <p className="mb-4">There was an error rendering the markdown content. Here&apos;s a basic version:</p>
                <div className="p-4 bg-gray-100 rounded overflow-auto whitespace-pre-wrap">
                  {sanitizedContent}
                </div>
              </div>
            );
          }
        })()}
      </div>
    </div>
  );
} 
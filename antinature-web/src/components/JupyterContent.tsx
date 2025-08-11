'use client';

import { JupyterContent } from '@/lib/jupyterUtils';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vs } from 'react-syntax-highlighter/dist/esm/styles/prism';
import SimplifiedMDXContent from './SimplifiedMDXContent';

interface JupyterContentViewProps {
  content: JupyterContent;
}

export function JupyterContentView({ content }: JupyterContentViewProps) {
  return (
    <div className="jupyter-notebook">
      <h1 className="mb-6 mt-10 text-3xl font-bold text-black">{content.title}</h1>
      
      {content.cells.map((cell, index) => (
        <div key={index} className="mb-6">
          {cell.cell_type === 'markdown' && (
            <div className="markdown-cell">
              <SimplifiedMDXContent content={Array.isArray(cell.source) ? cell.source.join('') : cell.source} />
            </div>
          )}
          
          {cell.cell_type === 'code' && (
            <div className="code-cell">
              <div className="code-input">
                <SyntaxHighlighter
                  language="python"
                  style={vs}
                  className="mb-2 rounded border border-gray-200"
                  codeTagProps={{
                    style: {
                      fontSize: '0.9rem',
                      fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
                    }
                  }}
                  customStyle={{
                    backgroundColor: 'white',
                    padding: '1rem'
                  }}
                >
                  {Array.isArray(cell.source) ? cell.source.join('') : cell.source}
                </SyntaxHighlighter>
              </div>
              
              {cell.outputs && cell.outputs.length > 0 && (
                <div className="notebook-output mt-2 border-l-4 border-gray-200 pl-3">
                  {cell.outputs.map((output, outputIndex) => (
                    <div key={outputIndex}>
                      {/* Text output */}
                      {output.output_type === 'stream' && output.text && (
                        <pre className="text-output bg-white border border-gray-200 p-2 text-sm text-gray-800 overflow-x-auto rounded">
                          {Array.isArray(output.text) ? output.text.join('') : output.text}
                        </pre>
                      )}
                      
                      {/* Display data (images, etc.) */}
                      {output.output_type === 'display_data' && output.data && (
                        <>
                          {/* Image output */}
                          {output.data['image/png'] && (
                            <img
                              src={`data:image/png;base64,${output.data['image/png']}`}
                              alt="Notebook output"
                              className="max-w-full mb-2 bg-white"
                            />
                          )}
                          
                          {/* HTML output */}
                          {output.data['text/html'] && (
                            <div 
                              className="bg-white p-2 border border-gray-200 rounded"
                              dangerouslySetInnerHTML={{ 
                                __html: Array.isArray(output.data['text/html']) 
                                  ? output.data['text/html'].join('') 
                                  : output.data['text/html'] 
                              }} 
                            />
                          )}
                          
                          {/* Plain text output */}
                          {output.data['text/plain'] && !output.data['text/html'] && !output.data['image/png'] && (
                            <pre className="text-output bg-white border border-gray-200 p-2 text-sm text-gray-800 overflow-x-auto rounded">
                              {Array.isArray(output.data['text/plain']) 
                                ? output.data['text/plain'].join('') 
                                : output.data['text/plain']}
                            </pre>
                          )}
                        </>
                      )}
                      
                      {/* Execute result (similar to display data) */}
                      {output.output_type === 'execute_result' && output.data && (
                        <>
                          {/* HTML output */}
                          {output.data['text/html'] && (
                            <div 
                              className="bg-white p-2 border border-gray-200 rounded"
                              dangerouslySetInnerHTML={{ 
                                __html: Array.isArray(output.data['text/html']) 
                                  ? output.data['text/html'].join('') 
                                  : output.data['text/html'] 
                              }} 
                            />
                          )}
                          
                          {/* Plain text output */}
                          {output.data['text/plain'] && !output.data['text/html'] && (
                            <pre className="text-output bg-white border border-gray-200 p-2 text-sm text-gray-800 overflow-x-auto rounded">
                              {Array.isArray(output.data['text/plain']) 
                                ? output.data['text/plain'].join('') 
                                : output.data['text/plain']}
                            </pre>
                          )}
                        </>
                      )}
                      
                      {/* Error output */}
                      {output.output_type === 'error' && (
                        <pre className="error-output bg-red-50 p-2 text-sm text-red-700 overflow-x-auto border border-red-200 rounded">
                          {output.traceback
                            ? Array.isArray(output.traceback)
                              ? output.traceback.join('')
                              : output.traceback
                            : `${output.ename}: ${output.evalue}`}
                        </pre>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
} 
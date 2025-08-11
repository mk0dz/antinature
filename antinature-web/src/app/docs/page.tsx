import { getContentBySlug } from '@/lib/contentUtils';
import { isMarkdownContent, isJupyterContent } from '@/lib/contentUtils';
import SimplifiedMDXContent from '@/components/SimplifiedMDXContent';
import { JupyterContentView } from '@/components/JupyterContent';

export default async function DocsPage() {
  // Load the overview content
  const content = await getContentBySlug('overview');
  
  if (!content) {
    return (
      <div className="prose">
        <h1>Documentation</h1>
        <p>Welcome to the Antinature documentation.</p>
        <p>Please select a topic from the sidebar to get started.</p>
      </div>
    );
  }
  
  return (
    <div className="prose prose-black max-w-none">
      {isMarkdownContent(content) && <SimplifiedMDXContent content={content.content} />}
      {isJupyterContent(content) && <JupyterContentView content={content} />}
    </div>
  );
}
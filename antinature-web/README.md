# Documentation System

This documentation system is designed to render markdown files and Jupyter notebooks from the `content` directory.

## Features

- Renders Markdown (`.md` and `.mdx`) files
- Renders Jupyter Notebooks (`.ipynb`) files
- Automatically generates navigation sidebar based on content structure
- Classic black and white theme with clean typography

## Content Structure

Place your documentation content in the `content` directory. The system will automatically parse the directory structure to create the navigation sidebar.

Example structure:
```
content/
├── getting-started.md
├── tutorials/
│   ├── tutorial1.md
│   └── tutorial2.ipynb
└── api/
    ├── endpoints.md
    └── authentication.md
```

## Markdown Frontmatter

You can use frontmatter to add metadata to your markdown files:

```md
---
title: Getting Started
description: Learn how to get started with our product
order: 1
---

# Getting Started

Content goes here...
```

The supported frontmatter fields are:
- `title`: Page title (used in navigation and page header)
- `description`: Short description (used in cards and meta tags)
- `order`: Number to sort items in the navigation (lower numbers appear first)

## Jupyter Notebooks

Jupyter notebooks are automatically converted to documentation pages. The system:

1. Extracts the title from the first markdown cell with a H1 heading
2. Renders markdown cells as-is
3. Formats code cells with syntax highlighting
4. Displays cell outputs (text and images)

## Development

To run the documentation system locally:

```bash
npm install
npm run dev
```

Then visit http://localhost:3000/docs

## Customization

- Edit `app/docs/layout.tsx` to modify the documentation layout
- Edit `components/MDXContent.tsx` to customize the rendering of markdown content
- Edit `components/Sidebar.tsx` to customize the navigation sidebar
- Edit `src/app/globals.css` to adjust the styling

## Dependencies

This system relies on:
- Next.js
- Tailwind CSS
- next-mdx-remote
- rehype/remark plugins for markdown processing

## Adding New Content

To add new content:

1. Create a new markdown or Jupyter notebook file in the `content` directory
2. The file will automatically appear in the navigation sidebar
3. Use frontmatter to control the title and ordering

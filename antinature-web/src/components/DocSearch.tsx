"use client";

import { useState, useEffect, useRef } from "react";
import { Search as SearchIcon } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";

interface SearchResult {
  title: string;
  href: string;
  excerpt: string;
}

export default function DocSearch() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();
  const searchRef = useRef<HTMLDivElement>(null);

  // Handle outside clicks
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [searchRef]);

  // Handle search
  useEffect(() => {
    if (!query) {
      setResults([]);
      setIsOpen(false);
      setError(null);
      return;
    }

    const delayDebounceFn = setTimeout(async () => {
      setIsLoading(true);
      setError(null);
      try {
        console.log(`Fetching search results for: ${query}`);
        const url = `/api/search?q=${encodeURIComponent(query)}`;
        console.log(`Request URL: ${url}`);
        
        const response = await fetch(url);
        console.log(`Response status: ${response.status}`);
        
        if (response.ok) {
          const data = await response.json();
          console.log(`Search results:`, data);
          setResults(data);
          setIsOpen(true);
        } else {
          const errorText = await response.text();
          console.error(`Search API error: ${response.status}`, errorText);
          setError(`Search failed: ${response.status}`);
          setResults([]);
        }
      } catch (error) {
        console.error("Search error:", error);
        setError(error instanceof Error ? error.message : "Search failed");
        setResults([]);
      } finally {
        setIsLoading(false);
      }
    }, 300);

    return () => clearTimeout(delayDebounceFn);
  }, [query]);

  // Handle keyboard navigation in search results
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && results.length > 0) {
      router.push(results[0].href);
      setIsOpen(false);
      setQuery("");
    } else if (e.key === "Escape") {
      setIsOpen(false);
    }
  };

  return (
    <div ref={searchRef} className="relative w-full">
      <SearchIcon className="absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-500" />
      <input
        type="search"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={handleKeyDown}
        onFocus={() => query && setIsOpen(true)}
        placeholder="Search documentation..."
        className="h-8 sm:h-10 w-full rounded-none border border-gray-300 bg-white pl-8 pr-4 text-xs sm:text-sm focus:border-gray-500 focus:outline-none"
      />

      {isOpen && (
        <div className="absolute top-full z-50 mt-1 w-full max-w-full rounded-md border border-gray-200 bg-white shadow-lg">
          {isLoading ? (
            <div className="p-2 sm:p-4 text-xs sm:text-sm text-gray-500">Searching...</div>
          ) : error ? (
            <div className="p-2 sm:p-4 text-xs sm:text-sm text-red-500">{error}</div>
          ) : results.length > 0 ? (
            <ul className="max-h-48 sm:max-h-60 md:max-h-80 overflow-auto">
              {results.map((result, index) => (
                <li key={index} className="border-b border-gray-100 last:border-0">
                  <Link
                    href={result.href}
                    className="block p-2 sm:p-3 hover:bg-gray-50"
                    onClick={() => {
                      setIsOpen(false);
                      setQuery("");
                    }}
                  >
                    <div className="text-xs sm:text-sm font-medium text-black">{result.title}</div>
                    <div className="mt-1 text-xs text-gray-500 hidden sm:block">{result.excerpt}</div>
                  </Link>
                </li>
              ))}
            </ul>
          ) : (
            <div className="p-2 sm:p-4 text-xs sm:text-sm text-gray-500">No results found</div>
          )}
        </div>
      )}
    </div>
  );
} 
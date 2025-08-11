import type React from "react"
import { Inter } from "next/font/google"
import Link from "next/link"

import "../globals.css"
import SidebarContainer from "@/components/SidebarContainer"
import DocSearch from "@/components/DocSearch"

const inter = Inter({ subsets: ["latin"] })

export const metadata = {
  title: "Antinature Documentation",
  description: "Documentation for the Antinature Python module",
}

export default function DocsLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className={`min-h-screen bg-white text-black ${inter.className}`}>
      <div className="flex flex-col md:flex-row min-h-screen">
        <div className="md:block">
          <SidebarContainer />
        </div>
        <div className="flex-1 border-l border-gray-200">
          <header className="sticky top-0 z-10 flex flex-col sm:flex-row sm:items-center justify-between border-b border-gray-200 bg-white px-3 sm:px-6 py-2 sm:py-0">
            <div className="w-full max-w-md mb-2 sm:mb-0 py-2">
              <DocSearch />
            </div>
            <nav className="flex items-center justify-between sm:justify-end space-x-1 xs:space-x-2 sm:space-x-6 sm:ml-auto py-1 sm:py-0">
              <Link href="/" className="text-xs sm:text-sm font-medium hover:text-gray-600 whitespace-nowrap px-1">
                Home
              </Link>
              <Link href="/docs" className="text-xs sm:text-sm font-medium hover:text-gray-600 whitespace-nowrap px-1">
                Documentation
              </Link>
              <Link
                href="https://github.com/mk0dz/antinature"
                target="_blank"
                className="text-xs sm:text-sm font-medium hover:text-gray-600 whitespace-nowrap px-1"
              >
                GitHub
              </Link>
            </nav>
          </header>
          <main className="pb-4 sm:pb-8 px-3 sm:px-6 md:px-8 pt-4 sm:pt-6">
            <div className="mx-auto max-w-full md:max-w-4xl">{children}</div>
          </main>
        </div>
      </div>
    </div>
  )
}


"use client"

import { useState } from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { ChevronDown, Menu, X } from "lucide-react"
import type { NavItem, NavigationMap } from "@/lib/contentUtils"

interface DocsSidebarProps {
  navItems?: NavItem[] | NavigationMap;
}

export default function DocsSidebar({ navItems = [] }: DocsSidebarProps) {
  const pathname = usePathname()
  const [sidebarOpen, setSidebarOpen] = useState(false)
  
  // Convert navigation map to array if needed
  const navItemsArray = Array.isArray(navItems) 
    ? navItems 
    : Object.entries(navItems as NavigationMap)
        .sort(([, a], [, b]) => a.order - b.order)
        .map(([key, section]) => ({
          title: section.title,
          slug: key.toLowerCase(),
          href: `/docs/${key.toLowerCase()}`,
          order: section.order,
          children: section.children || []
        }));
  
  const [openSections, setOpenSections] = useState<Record<string, boolean>>(() => {
    // Initialize open state for sections based on current path
    const initialOpenSections: Record<string, boolean> = {}
    
    // Auto-expand sections based on current path
    if (navItemsArray && navItemsArray.length > 0) {
      navItemsArray.forEach(item => {
        if (item.children && item.children.length > 0) {
          // Check if current path matches any child
          const isActive = item.children.some(
            (child: NavItem) => pathname === child.href || pathname.startsWith(child.href + '/')
          )
          // Always open tutorials and examples sections for better discoverability
          // Also open any section that contains the current page
          initialOpenSections[item.title] = isActive || 
                                         item.title === 'Tutorials' || 
                                         item.title === 'Examples'
        }
      })
    }
    
    return initialOpenSections
  })

  const toggleSection = (title: string) => {
    setOpenSections((prev) => ({
      ...prev,
      [title]: !prev[title],
    }))
  }

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen)
  }

  const renderNavItems = (items: NavItem[], level = 0) => {
    return items.map((item, index) => {
      // Ensure we have a truly unique key using multiple properties
      const uniqueKey = `${item.title}-${item.href}-${index}`;
      const isActive = pathname === item.href
      const isActiveParent = pathname.startsWith(item.href + '/')
      const hasChildren = item.children && item.children.length > 0
      const isOpen = openSections[item.title]

      return (
        <div key={uniqueKey} className={`mb-1 ${level > 0 ? "ml-4" : ""}`}>
          <div className="flex items-center">
            {hasChildren ? (
              <button
                onClick={() => toggleSection(item.title)}
                className={`flex w-full items-center justify-between py-1 text-left text-sm hover:text-gray-600 ${
                  isActive || isActiveParent ? "font-medium text-black" : "text-gray-700"
                }`}
              >
                {item.title}
                <ChevronDown className={`h-4 w-4 transition-transform ${isOpen ? "rotate-180 transform" : ""}`} />
              </button>
            ) : (
              <Link
                href={item.href}
                className={`block w-full py-1 text-sm hover:text-gray-600 ${
                  isActive ? "font-medium text-black" : "text-gray-700"
                }`}
                onClick={() => setSidebarOpen(false)}
              >
                {item.title}
              </Link>
            )}
          </div>
          {hasChildren && isOpen && <div className="mt-1">{renderNavItems(item.children || [], level + 1)}</div>}
        </div>
      )
    })
  }

  return (
    <>
      {/* Mobile menu button */}
      <div className="md:hidden fixed top-0 left-0 z-40 mt-3 ml-3">
        <button 
          onClick={toggleSidebar}
          className="p-2 text-gray-600 hover:text-gray-900 focus:outline-none"
          aria-label="Toggle sidebar"
        >
          {sidebarOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>
      
      {/* Overlay for mobile */}
      {sidebarOpen && (
        <div 
          className="md:hidden fixed inset-0 bg-black bg-opacity-50 z-30"
          onClick={() => setSidebarOpen(false)}
        ></div>
      )}
      
      <div 
        className={`
          fixed md:sticky top-0 z-30 
          h-screen w-64 shrink-0 bg-white 
          overflow-y-auto
          transition-transform duration-300 ease-in-out
          md:translate-x-0 
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
          shadow-lg md:shadow-none
        `}
      >
        <div className="px-4 py-6">
          <div className="mb-6 border-b border-gray-200 pb-4 flex justify-between items-center">
            <div>
              <h2 className="text-lg font-bold">Antinature </h2>
              <p className="mt-1 text-xs text-gray-500">0.1.0</p>
            </div>
            <button 
              onClick={toggleSidebar}
              className="md:hidden p-1 text-gray-500 hover:text-gray-700"
              aria-label="Close sidebar"
            >
              <X size={20} />
            </button>
          </div>
          <nav className="space-y-1">
            {navItemsArray.length > 0 ? (
              renderNavItems(navItemsArray)
            ) : (
              <div className="py-2 text-sm text-gray-500">Loading navigation...</div>
            )}
          </nav>
        </div>
      </div>
    </>
  )
}


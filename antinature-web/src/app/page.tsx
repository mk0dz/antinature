'use client'
import Image from 'next/image'
import Link from 'next/link'
import React, { useState } from 'react';

export default function Home() {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText('pip install antinature');
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  
  return (
    <div className="relative flex flex-col-reverse md:flex-row h-screen bg-white text-black ">

      {/* Hero Section (Right Side on Desktop, Top on Mobile) */}
      <div className="w-full md:w-2/3 p-6 sm:p-10 md:p-16 lg:p-20 overflow-auto relative">
        <link href="https://fonts.googleapis.com/css2?family=Chivo+Mono:ital,wght@0,100..900;1,100..900&display=swap" rel="stylesheet"></link>
        <link href="https://fonts.googleapis.com/css2?family=Wix+Madefor+Display:wght@400..800&display=swap" rel="stylesheet"></link>
        <h1 style={{fontFamily: 'Wix Madefor Display'}} className="relative text-4xl sm:text-5xl md:text-6xl font-thin mb-3 sm:mb-6">Dirac&apos;s</h1>
        <p style={{fontFamily: 'Wix Madefor Display'}} className="relative text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-thin mb-2 sm:mb-4">
         Antinature
        </p>
        <p style={{fontFamily: 'Chivo Mono'}} className='text-xl sm:text-2xl md:text-3xl lg:text-4xl tracking-wide max-w-2xl font-thin mb-2 sm:mb-4 p-0 sm:p-2 md:p-4'>
          py module for playing with antimatter by using quantum computing approach
        </p>
        <div className="mb-2 sm:mb-4 mt-4 sm:mt-8 flex flex-col space-y-4 sm:flex-row sm:space-x-4 sm:space-y-0">
          <Link
            href="/docs/getstarted"
            className="inline-flex items-center justify-center border border-black bg-black px-4 sm:px-6 py-2 sm:py-3 text-sm font-medium text-white hover:bg-gray-800"
          >
            Get Started
          </Link>
          <Link
            href="/docs"
            className="inline-flex items-center justify-center border border-gray-300 bg-white px-4 sm:px-6 py-2 sm:py-3 text-sm font-medium text-black hover:bg-gray-50"
          >
            Documentation
          </Link>
        </div>
        <div className="relative overflow-x-auto mt-4" >
          <pre className="inline-flex items-center justify-center border border-gray-300 bg-white px-3 sm:px-6 py-2 sm:py-3 text-base sm:text-xl font-medium text-black hover:bg-gray-50 max-w-full sm:max-w-sm relative pr-10 sm:pr-12 flex items-center gap-2 overflow-x-auto">
            <code>pip install antinature</code>
            <button 
              onClick={handleCopy}
              className="absolute right-1 sm:right-3 top-1/2 transform -translate-y-1/2 p-1 sm:p-2 transition-all duration-300 text-gray-300 hover:text-white hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-400 rounded-md"
              aria-label="Copy to clipboard"
              title="Copy to clipboard"
            >
              {copied ? (
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 sm:h-6 sm:w-6" viewBox="0 0 20 20" fill="currentColor">
                  <path d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9z" />
                  <path d="M7 8a1 1 0 012 0v.5a1 1 0 01-1 1H7v1a1 1 0 001 1h.5a1 1 0 010 2h-2a1 1 0 01-1-1v-5a1 1 0 011-1h.5z" />
                  <path d="M12 8a1 1 0 00-1 1v.5h1.5a1 1 0 110 2H11v1a1 1 0 102 0v-3.5a1 1 0 00-1-1z" />
                </svg>
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 sm:h-6 sm:w-6" viewBox="0 0 20 20" fill="currentColor">
                  <path d="M8 2a1 1 0 000 2h2a1 1 0 100-2H8z" />
                  <path d="M3 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v6h-4.586l1.293-1.293a1 1 0 00-1.414-1.414l-3 3a1 1 0 000 1.414l3 3a1 1 0 001.414-1.414L10.414 13H15v3a2 2 0 01-2 2H5a2 2 0 01-2-2V5z" />
                </svg>
              )}
            </button>
          </pre>
          {copied && (
            <div className="absolute right-10 sm:right-16 top-1/2 transform -translate-y-1/2 bg-gray-800 text-white px-2 sm:px-3 py-1 rounded-md animate-fade-in-out text-xs sm:text-sm">
              Copied!
            </div>
          )}
        </div>
       
        <pre className="overflow-x-auto relative inline-flex items-center mt-4 justify-center border border-gray-300 bg-white px-3 sm:px-6 py-2 sm:py-3 font-medium text-black hover:bg-gray-50 md:px-4 md:py-2 w-full md:w-auto">
            <code className='relative text-sm sm:text-base md:text-lg lg:text-xl tracking-wide'>
                from antinature.core import MixedMatterBasis <br />
                from antinature.core import AntinatureHamiltonian <br />
                from antinature.core import AntinatureIntegralEngine <br />
                from antinature.specialized import PositroniumSCF <br />
                from antinature.core import MolecularData <br />
            </code>
            
        </pre>
        <Image 
          src="/qiskitcite.svg"
          alt="Paul Dirac"
          width={290}
          height={29}
          className='items-end justify-end p-2 mt-2 max-w-full'
        />
      </div>
      

      {/* Paul Dirac Image (Left Side on Desktop, Bottom on Mobile) */}
      <div className="relative w-full md:w-1/3 h-64 md:h-auto">
      
        <Image 
          src="/dirac.png"
          alt="Paul Dirac"
          layout="fill"
          objectFit="cover"
        />
        <p style={{fontFamily: 'Chivo Mono'}} className='absolute bottom-0 left-2 text-xs sm:text-sm md:text-md font-thin text-white bg-black bg-opacity-30 px-2 py-1 rounded'>Copyright © 2025, <a href="https://github.com/mk0dz" className='underline'>mk0dz</a></p>
      </div>

    </div>
  )
}




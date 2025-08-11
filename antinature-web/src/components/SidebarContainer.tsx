import { generateNavigation } from "@/lib/contentUtils"
import DocsSidebar from "./sidebar"

export default async function SidebarContainer() {
  try {
    // Fetch navigation on the server
    const navItems = await generateNavigation()
    
    // Pass data to the client component
    return <DocsSidebar navItems={navItems} />
  } catch (error) {
    console.error("Error generating navigation:", error);
    // Return sidebar with empty navigation in case of error
    return <DocsSidebar navItems={[]} />
  }
} 
import { useState } from "react";
import { Search, Scale, AlertTriangle, TrendingUp, ChevronRight, Clock, Gavel, BookOpen } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { useNavigate } from "react-router-dom";

const Dashboard = () => {
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState("");

  const featuredCase = {
    id: "2",
    title: "Tech Corp. v. Innovation LLC",
    court: "Ninth Circuit Court of Appeals",
    date: "Nov 10, 2024",
    excerpt: "The court held that algorithmic trade secrets are subject to different standards of protection under the DTSA when the algorithm's output is publicly observable...",
    tags: ["Trade Secrets", "Technology", "Appeals"],
    contradictions: 3,
    citations: 47,
  };

  const recentCases = [
    { id: "1", title: "Smith v. Jones", court: "Supreme Court", jurisdiction: "Federal", status: "89% match" },
    { id: "3", title: "State v. Johnson", court: "District Court", jurisdiction: "CA", status: "92% match" },
    { id: "case-001", title: "Williams v. DataCorp", court: "Court of Appeals", jurisdiction: "NY", status: "High precedent" },
  ];

  return (
    <div className="min-h-screen bg-background">
      {/* Masthead */}
      <div className="border-b-4 border-legal-navy bg-background">
        <div className="container mx-auto px-6 py-6 max-w-7xl">
          <div className="flex items-end justify-between">
            <div>
              <div className="flex items-center gap-3 mb-1">
                <Scale className="h-8 w-8 text-legal-navy" />
                <h1 className="text-5xl font-serif font-bold text-legal-navy tracking-tight">LexLink</h1>
              </div>
              <p className="text-sm uppercase tracking-widest text-muted-foreground font-mono">
                Intelligent Legal Research · Knowledge Graphs · AI Analysis
              </p>
            </div>
            <div className="text-right">
              <div className="text-xs uppercase tracking-wider text-muted-foreground mb-1">Today</div>
              <div className="text-lg font-serif font-bold text-foreground">November 23, 2025</div>
            </div>
          </div>
        </div>
      </div>

      {/* Search Hero */}
      <div className="bg-legal-navy text-white">
        <div className="container mx-auto px-6 py-12 max-w-7xl">
          <div className="max-w-3xl">
            <h2 className="text-3xl font-serif font-bold mb-4">Search the Knowledge Graph</h2>
            <div className="relative">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-white/60" />
              <input
                type="text"
                placeholder="Enter case name, citation, or legal concept..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full h-14 pl-12 pr-4 bg-white/10 border-2 border-white/20 rounded-none text-white placeholder:text-white/50 focus:bg-white/15 focus:border-legal-gold focus:outline-none text-lg transition-all"
              />
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-12 max-w-7xl">
        {/* Main Grid Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-12">
          {/* Featured Case - Takes 2 columns */}
          <div className="lg:col-span-2">
            <div className="mb-2 flex items-center gap-2">
              <div className="h-1 w-12 bg-legal-gold"></div>
              <span className="text-xs uppercase tracking-widest font-mono text-muted-foreground">Featured Analysis</span>
            </div>
            <div 
              className="border-l-8 border-legal-gold bg-card p-8 cursor-pointer hover:shadow-2xl transition-shadow group"
              onClick={() => navigate(`/case/${featuredCase.id}`)}
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                  <h2 className="text-3xl font-serif font-bold text-foreground mb-2 group-hover:text-legal-blue transition-colors">
                    {featuredCase.title}
                  </h2>
                  <div className="flex items-center gap-3 text-sm text-muted-foreground">
                    <span className="font-medium">{featuredCase.court}</span>
                    <span>·</span>
                    <span>{featuredCase.date}</span>
                  </div>
                </div>
                {featuredCase.contradictions > 0 && (
                  <Badge variant="destructive" className="gap-1">
                    <AlertTriangle className="h-3 w-3" />
                    {featuredCase.contradictions} Conflicts
                  </Badge>
                )}
              </div>
              
              <p className="text-foreground/80 mb-6 leading-relaxed text-lg">
                {featuredCase.excerpt}
              </p>

              <div className="flex items-center justify-between">
                <div className="flex gap-2">
                  {featuredCase.tags.map((tag) => (
                    <Badge key={tag} variant="secondary" className="rounded-none font-mono text-xs">
                      {tag}
                    </Badge>
                  ))}
                </div>
                <div className="flex items-center gap-4 text-sm text-muted-foreground">
                  <span>{featuredCase.citations} citations</span>
                  <ChevronRight className="h-4 w-4" />
                </div>
              </div>
            </div>
          </div>

          {/* Stats Column */}
          <div className="space-y-6">
            <div className="mb-2 flex items-center gap-2">
              <div className="h-1 w-12 bg-legal-blue"></div>
              <span className="text-xs uppercase tracking-widest font-mono text-muted-foreground">System Metrics</span>
            </div>

            <div className="bg-legal-navy text-white p-6">
              <div className="text-4xl font-serif font-bold mb-1">24,891</div>
              <div className="text-sm uppercase tracking-wider text-white/70">Total Cases Indexed</div>
            </div>

            <div className="bg-card border-2 border-destructive p-6">
              <div className="text-4xl font-serif font-bold text-destructive mb-1">342</div>
              <div className="text-sm uppercase tracking-wider text-muted-foreground">Contradictions Detected</div>
            </div>

            <div className="bg-card border-2 border-success p-6">
              <div className="text-4xl font-serif font-bold text-success mb-1">89,432</div>
              <div className="text-sm uppercase tracking-wider text-muted-foreground">Graph Nodes</div>
            </div>

            <div className="bg-legal-gold/10 border-2 border-legal-gold p-6">
              <div className="text-4xl font-serif font-bold text-legal-gold mb-1">76%</div>
              <div className="text-sm uppercase tracking-wider text-muted-foreground">Research Time Saved</div>
            </div>
          </div>
        </div>

        {/* Recent Cases - Newspaper Style */}
        <div className="mb-12">
          <div className="mb-4 flex items-center gap-2">
            <div className="h-1 w-12 bg-legal-blue"></div>
            <span className="text-xs uppercase tracking-widest font-mono text-muted-foreground">Recent Activity</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {recentCases.map((case_) => (
              <div
                key={case_.id}
                onClick={() => navigate(`/case/${case_.id}`)}
                className="bg-card border-l-4 border-legal-blue p-6 cursor-pointer hover:shadow-lg transition-all group"
              >
                <div className="text-xs uppercase tracking-wider text-muted-foreground mb-2 font-mono">
                  {case_.jurisdiction}
                </div>
                <h3 className="text-xl font-serif font-bold mb-2 group-hover:text-legal-blue transition-colors">
                  {case_.title}
                </h3>
                <p className="text-sm text-muted-foreground mb-3">{case_.court}</p>
                <div className="flex items-center justify-between">
                  <Badge variant="secondary" className="rounded-none text-xs">
                    {case_.status}
                  </Badge>
                  <ChevronRight className="h-4 w-4 text-muted-foreground" />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Tools Section - Horizontal Blocks */}
        <div className="mb-4 flex items-center gap-2">
          <div className="h-1 w-12 bg-legal-gold"></div>
          <span className="text-xs uppercase tracking-widest font-mono text-muted-foreground">Research Tools</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-1">
          <div 
            onClick={() => navigate("/case/case-001")}
            className="bg-legal-navy text-white p-8 cursor-pointer hover:bg-legal-blue transition-colors group"
          >
            <BookOpen className="h-10 w-10 mb-4 text-legal-gold" />
            <h3 className="text-2xl font-serif font-bold mb-2">Document Summarization</h3>
            <p className="text-white/70 text-sm mb-4">
              AI-powered summaries with Legal-BERT, preserving citations and reasoning
            </p>
            <div className="flex items-center gap-2 text-legal-gold font-mono text-sm">
              <span>Analyze</span>
              <ChevronRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
            </div>
          </div>

          <div 
            onClick={() => navigate("/knowledge-graph")}
            className="bg-success text-white p-8 cursor-pointer hover:bg-success/90 transition-colors group"
          >
            <Gavel className="h-10 w-10 mb-4 text-white" />
            <h3 className="text-2xl font-serif font-bold mb-2">Knowledge Graph</h3>
            <p className="text-white/90 text-sm mb-4">
              Explore precedent chains and case relationships via Neo4j
            </p>
            <div className="flex items-center gap-2 text-white font-mono text-sm">
              <span>Explore</span>
              <ChevronRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
            </div>
          </div>

          <div 
            onClick={() => navigate("/case/case-002")}
            className="bg-destructive text-white p-8 cursor-pointer hover:bg-destructive/90 transition-colors group"
          >
            <AlertTriangle className="h-10 w-10 mb-4 text-white" />
            <h3 className="text-2xl font-serif font-bold mb-2">Contradiction Detection</h3>
            <p className="text-white/90 text-sm mb-4">
              Identify conflicting rulings across jurisdictions using NLI models
            </p>
            <div className="flex items-center gap-2 text-white font-mono text-sm">
              <span>Detect</span>
              <ChevronRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;

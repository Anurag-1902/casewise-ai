import { useState } from "react";
import { Search, FileText, Network, AlertTriangle, TrendingUp } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useNavigate } from "react-router-dom";

const Dashboard = () => {
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState("");

  const stats = [
    { label: "Total Cases", value: "24,891", icon: FileText, color: "text-legal-blue" },
    { label: "Contradictions Found", value: "342", icon: AlertTriangle, color: "text-warning" },
    { label: "Knowledge Graph Nodes", value: "89,432", icon: Network, color: "text-success" },
    { label: "Research Time Saved", value: "76%", icon: TrendingUp, color: "text-legal-gold" },
  ];

  const recentCases = [
    {
      id: "1",
      title: "Smith v. Jones",
      court: "Supreme Court",
      date: "2024-11-15",
      similarity: 0.89,
      status: "Similar to 12 cases",
    },
    {
      id: "2",
      title: "Tech Corp. v. Innovation LLC",
      court: "Court of Appeals",
      date: "2024-11-10",
      similarity: 0.76,
      status: "Contradiction detected",
      hasContradiction: true,
    },
    {
      id: "3",
      title: "State v. Johnson",
      court: "District Court",
      date: "2024-11-08",
      similarity: 0.92,
      status: "High precedent value",
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="h-12 w-12 rounded-lg bg-gradient-legal flex items-center justify-center">
              <Network className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold text-foreground">LexLink</h1>
              <p className="text-muted-foreground">Intelligent Knowledge Graphs for Legal Research</p>
            </div>
          </div>
        </div>

        {/* Search Bar */}
        <Card className="mb-8 border-2 shadow-lg">
          <CardContent className="pt-6">
            <div className="flex gap-3">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
                <Input
                  placeholder="Search cases, statutes, or legal concepts..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10 h-12 text-lg"
                />
              </div>
              <Button size="lg" className="px-8 bg-gradient-legal hover:opacity-90">
                Search
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {stats.map((stat) => (
            <Card key={stat.label} className="hover:shadow-lg transition-shadow">
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  {stat.label}
                </CardTitle>
                <stat.icon className={`h-5 w-5 ${stat.color}`} />
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{stat.value}</div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Recent Cases */}
        <Card className="shadow-lg">
          <CardHeader>
            <CardTitle className="text-2xl">Recent Case Analysis</CardTitle>
            <CardDescription>
              Latest cases processed through semantic analysis and contradiction detection
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {recentCases.map((case_) => (
                <div
                  key={case_.id}
                  className="flex items-center justify-between p-4 rounded-lg border bg-card hover:bg-accent/5 transition-colors cursor-pointer"
                  onClick={() => navigate(`/case/${case_.id}`)
                  }
                >
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <h3 className="font-semibold text-lg">{case_.title}</h3>
                      {case_.hasContradiction && (
                        <Badge variant="destructive" className="gap-1">
                          <AlertTriangle className="h-3 w-3" />
                          Contradiction
                        </Badge>
                      )}
                    </div>
                    <div className="flex items-center gap-4 text-sm text-muted-foreground">
                      <span>{case_.court}</span>
                      <span>•</span>
                      <span>{case_.date}</span>
                      <span>•</span>
                      <span className="text-success font-medium">
                        {Math.round(case_.similarity * 100)}% similarity score
                      </span>
                    </div>
                  </div>
                  <div className="text-right">
                    <Badge variant="secondary">{case_.status}</Badge>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
          <Card 
            className="hover:shadow-lg transition-shadow cursor-pointer border-l-4 border-l-legal-blue"
            onClick={() => navigate("/case/case-001")}
          >
            <CardHeader>
              <FileText className="h-8 w-8 text-legal-blue mb-2" />
              <CardTitle>Document Summarization</CardTitle>
              <CardDescription>
                Generate AI-powered summaries preserving legal reasoning and citations
              </CardDescription>
            </CardHeader>
          </Card>

          <Card 
            className="hover:shadow-lg transition-shadow cursor-pointer border-l-4 border-l-success"
            onClick={() => navigate("/knowledge-graph")}
          >
            <CardHeader>
              <Network className="h-8 w-8 text-success mb-2" />
              <CardTitle>Knowledge Graph</CardTitle>
              <CardDescription>
                Explore case relationships and precedent chains using Neo4j
              </CardDescription>
            </CardHeader>
          </Card>

          <Card 
            className="hover:shadow-lg transition-shadow cursor-pointer border-l-4 border-l-warning"
            onClick={() => navigate("/case/case-002")}
          >
            <CardHeader>
              <AlertTriangle className="h-8 w-8 text-warning mb-2" />
              <CardTitle>Contradiction Detection</CardTitle>
              <CardDescription>
                Identify conflicting rulings across jurisdictions and time periods
              </CardDescription>
            </CardHeader>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;

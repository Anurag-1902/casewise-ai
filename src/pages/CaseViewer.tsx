import { useState } from "react";
import { ArrowLeft, FileText, Scale, Calendar, MapPin, Users, AlertTriangle, Link } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { mockCases } from "@/data/mockCases";
import { useNavigate, useParams } from "react-router-dom";

const CaseViewer = () => {
  const navigate = useNavigate();
  const { id } = useParams();
  const caseData = mockCases.find(c => c.id === id) || mockCases[0];
  const [activeTab, setActiveTab] = useState("summary");

  const similarCases = mockCases.filter(c => caseData.similarCases.includes(c.id));
  const contradictoryCases = mockCases.filter(c => caseData.contradictions.includes(c.id));

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="mb-6">
          <Button
            variant="ghost"
            onClick={() => navigate("/")}
            className="mb-4"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Dashboard
          </Button>

          <div className="flex items-start justify-between">
            <div className="flex-1">
              <h1 className="text-4xl font-bold mb-2">{caseData.title}</h1>
              <p className="text-xl text-muted-foreground mb-4">{caseData.citation}</p>
              
              <div className="flex flex-wrap gap-3">
                <div className="flex items-center gap-2 text-sm">
                  <Scale className="h-4 w-4 text-legal-blue" />
                  <span>{caseData.court}</span>
                </div>
                <Separator orientation="vertical" className="h-5" />
                <div className="flex items-center gap-2 text-sm">
                  <Calendar className="h-4 w-4 text-legal-blue" />
                  <span>{caseData.date}</span>
                </div>
                <Separator orientation="vertical" className="h-5" />
                <div className="flex items-center gap-2 text-sm">
                  <MapPin className="h-4 w-4 text-legal-blue" />
                  <span>{caseData.jurisdiction}</span>
                </div>
              </div>

              <div className="flex flex-wrap gap-2 mt-4">
                {caseData.categories.map((cat) => (
                  <Badge key={cat} variant="secondary">
                    {cat}
                  </Badge>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-6">
            <Card className="shadow-lg">
              <Tabs value={activeTab} onValueChange={setActiveTab}>
                <CardHeader>
                  <TabsList className="grid w-full grid-cols-3">
                    <TabsTrigger value="summary">AI Summary</TabsTrigger>
                    <TabsTrigger value="full">Full Text</TabsTrigger>
                    <TabsTrigger value="analysis">Analysis</TabsTrigger>
                  </TabsList>
                </CardHeader>
                <CardContent>
                  <TabsContent value="summary" className="space-y-4">
                    <div className="prose prose-sm max-w-none">
                      <div className="bg-accent/10 border-l-4 border-l-legal-gold p-4 rounded-r-lg mb-4">
                        <p className="text-sm font-semibold text-legal-gold mb-1">
                          AI-Generated Summary (Legal-BERT + BART)
                        </p>
                        <p className="text-sm text-muted-foreground">
                          Preserves legal reasoning, holdings, and citations
                        </p>
                      </div>
                      <p className="text-foreground leading-relaxed">{caseData.summary}</p>
                    </div>
                  </TabsContent>

                  <TabsContent value="full" className="space-y-4">
                    <div className="prose prose-sm max-w-none">
                      <pre className="whitespace-pre-wrap font-serif text-sm leading-relaxed">
                        {caseData.fullText}
                      </pre>
                    </div>
                  </TabsContent>

                  <TabsContent value="analysis" className="space-y-4">
                    <div className="space-y-4">
                      <div>
                        <h3 className="font-semibold mb-2 flex items-center gap-2">
                          <Users className="h-4 w-4 text-legal-blue" />
                          Panel / Authorship
                        </h3>
                        <div className="flex flex-wrap gap-2">
                          {caseData.judges.map((judge) => (
                            <Badge key={judge} variant="outline">
                              {judge}
                            </Badge>
                          ))}
                        </div>
                      </div>

                      <Separator />

                      <div>
                        <h3 className="font-semibold mb-2 flex items-center gap-2">
                          <Link className="h-4 w-4 text-legal-blue" />
                          Citations Referenced
                        </h3>
                        <ul className="space-y-2">
                          {caseData.citations.map((cite, idx) => (
                            <li key={idx} className="text-sm text-muted-foreground pl-4 border-l-2 border-muted">
                              {cite}
                            </li>
                          ))}
                        </ul>
                      </div>

                      <Separator />

                      <div className="bg-muted/30 p-4 rounded-lg">
                        <h3 className="font-semibold mb-2">Legal-BERT Embedding</h3>
                        <p className="text-sm text-muted-foreground">
                          768-dimensional vector generated for semantic similarity analysis
                        </p>
                        <code className="text-xs mt-2 block">
                          [0.234, -0.891, 0.456, 0.123, -0.678, ...]
                        </code>
                      </div>
                    </div>
                  </TabsContent>
                </CardContent>
              </Tabs>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Similar Cases */}
            {similarCases.length > 0 && (
              <Card className="shadow-lg">
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <FileText className="h-5 w-5 text-success" />
                    Similar Cases
                  </CardTitle>
                  <CardDescription>
                    FAISS similarity search (cosine ≥ 0.75)
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  {similarCases.map((case_) => (
                    <div
                      key={case_.id}
                      className="p-3 rounded-lg border bg-card hover:bg-accent/5 transition-colors cursor-pointer"
                      onClick={() => navigate(`/case/${case_.id}`)}
                    >
                      <h4 className="font-semibold text-sm mb-1">{case_.title}</h4>
                      <p className="text-xs text-muted-foreground mb-2">{case_.citation}</p>
                      <div className="flex items-center justify-between">
                        <Badge variant="secondary" className="text-xs">
                          {case_.court}
                        </Badge>
                        <span className="text-xs font-medium text-success">
                          89% similar
                        </span>
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            )}

            {/* Contradictions */}
            {contradictoryCases.length > 0 && (
              <Card className="shadow-lg border-destructive/50">
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <AlertTriangle className="h-5 w-5 text-destructive" />
                    Contradictory Rulings
                  </CardTitle>
                  <CardDescription>
                    NLI-based contradiction detection
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  {contradictoryCases.map((case_) => (
                    <div
                      key={case_.id}
                      className="p-3 rounded-lg border border-destructive/30 bg-destructive/5 hover:bg-destructive/10 transition-colors cursor-pointer"
                      onClick={() => navigate(`/case/${case_.id}`)}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <h4 className="font-semibold text-sm">{case_.title}</h4>
                        <Badge variant="destructive" className="text-xs">
                          Conflict
                        </Badge>
                      </div>
                      <p className="text-xs text-muted-foreground mb-2">{case_.citation}</p>
                      <p className="text-xs">
                        <span className="font-semibold">Type:</span> Jurisdictional Divergence
                      </p>
                      <p className="text-xs mt-1">
                        <span className="font-semibold">Issue:</span> Arbitration disclosure requirements
                      </p>
                    </div>
                  ))}
                </CardContent>
              </Card>
            )}

            {/* Metadata */}
            <Card className="shadow-lg">
              <CardHeader>
                <CardTitle className="text-lg">Case Metadata</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <div>
                  <span className="font-semibold">Processed:</span>
                  <p className="text-muted-foreground">2024-11-22 14:32 UTC</p>
                </div>
                <Separator />
                <div>
                  <span className="font-semibold">Knowledge Graph:</span>
                  <p className="text-muted-foreground">Indexed with 23 relationships</p>
                </div>
                <Separator />
                <div>
                  <span className="font-semibold">Model Version:</span>
                  <p className="text-muted-foreground">Legal-BERT v2.1</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CaseViewer;

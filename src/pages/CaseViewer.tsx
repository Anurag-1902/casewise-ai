import { useState } from "react";
import { ArrowLeft, FileText, Scale, Calendar, MapPin, Users, AlertTriangle, Link } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { useNavigate, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";

const CaseViewer = () => {
  const navigate = useNavigate();
  const { id } = useParams();
  const [activeTab, setActiveTab] = useState("summary");

  const { data: caseData, isLoading } = useQuery({
    queryKey: ['legal-case', id],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('legal_cases')
        .select('*')
        .eq('id', id)
        .maybeSingle();
      
      if (error) throw error;
      return data;
    },
  });

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20">
        <div className="container mx-auto px-4 py-8 max-w-7xl">
          <div className="animate-pulse space-y-6">
            <div className="h-8 bg-muted rounded w-1/4"></div>
            <div className="h-12 bg-muted rounded w-3/4"></div>
            <div className="h-96 bg-muted rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  if (!caseData) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20">
        <div className="container mx-auto px-4 py-8 max-w-7xl">
          <Button variant="ghost" onClick={() => navigate("/")} className="mb-4">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Dashboard
          </Button>
          <div className="text-center py-12">
            <h2 className="text-2xl font-bold mb-2">Case Not Found</h2>
            <p className="text-muted-foreground">The requested case could not be found in the database.</p>
          </div>
        </div>
      </div>
    );
  }

  const citationsArray = Array.isArray(caseData.citations) ? caseData.citations : [];
  const analysisData = caseData.analysis as any;
  const judgesList = analysisData?.judges || [];

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
              <h1 className="text-4xl font-bold mb-2">{caseData.name}</h1>
              <p className="text-xl text-muted-foreground mb-4">{caseData.citation || caseData.name_abbreviation}</p>
              
              <div className="flex flex-wrap gap-3">
                <div className="flex items-center gap-2 text-sm">
                  <Scale className="h-4 w-4 text-legal-blue" />
                  <span>{caseData.court || 'Unknown Court'}</span>
                </div>
                <Separator orientation="vertical" className="h-5" />
                <div className="flex items-center gap-2 text-sm">
                  <Calendar className="h-4 w-4 text-legal-blue" />
                  <span>{caseData.decision_date ? new Date(caseData.decision_date).toLocaleDateString() : 'Date unknown'}</span>
                </div>
                <Separator orientation="vertical" className="h-5" />
                <div className="flex items-center gap-2 text-sm">
                  <MapPin className="h-4 w-4 text-legal-blue" />
                  <span>{caseData.jurisdiction || 'N/A'}</span>
                </div>
              </div>

              {caseData.docket_number && (
                <div className="mt-4">
                  <Badge variant="secondary">
                    Docket: {caseData.docket_number}
                  </Badge>
                </div>
              )}
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
                          Case Preview
                        </p>
                        <p className="text-sm text-muted-foreground">
                          Summary and key details from Case.Law database
                        </p>
                      </div>
                      <div className="text-foreground leading-relaxed space-y-2">
                        {caseData.preview && caseData.preview.length > 0 ? (
                          caseData.preview.map((text: string, idx: number) => (
                            <p key={idx}>{text}</p>
                          ))
                        ) : (
                          <p>No preview available for this case.</p>
                        )}
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="full" className="space-y-4">
                    <div className="prose prose-sm max-w-none">
                      {caseData.url ? (
                        <div className="space-y-4">
                          <p className="text-muted-foreground">
                            Full case text is available at Case.Law. Click the link below to view:
                          </p>
                          <a 
                            href={caseData.url} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-2 text-legal-blue hover:underline"
                          >
                            View full text on Case.Law
                            <Link className="h-4 w-4" />
                          </a>
                        </div>
                      ) : (
                        <p className="text-muted-foreground">Full text not available for this case.</p>
                      )}
                    </div>
                  </TabsContent>

                  <TabsContent value="analysis" className="space-y-4">
                    <div className="space-y-4">
                      {judgesList.length > 0 && (
                        <>
                          <div>
                            <h3 className="font-semibold mb-2 flex items-center gap-2">
                              <Users className="h-4 w-4 text-legal-blue" />
                              Panel / Authorship
                            </h3>
                            <div className="flex flex-wrap gap-2">
                              {judgesList.map((judge: string, idx: number) => (
                                <Badge key={idx} variant="outline">
                                  {judge}
                                </Badge>
                              ))}
                            </div>
                          </div>
                          <Separator />
                        </>
                      )}

                      <div>
                        <h3 className="font-semibold mb-2 flex items-center gap-2">
                          <Link className="h-4 w-4 text-legal-blue" />
                          Citations Referenced
                        </h3>
                        {citationsArray.length > 0 ? (
                          <ul className="space-y-2">
                            {citationsArray.map((cite: any, idx: number) => (
                              <li key={idx} className="text-sm text-muted-foreground pl-4 border-l-2 border-muted">
                                {typeof cite === 'string' ? cite : cite.cite || JSON.stringify(cite)}
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <p className="text-sm text-muted-foreground">No citations available</p>
                        )}
                      </div>

                      <Separator />

                      <div className="bg-muted/30 p-4 rounded-lg">
                        <h3 className="font-semibold mb-2">Case Metadata</h3>
                        <div className="space-y-2 text-sm">
                          {caseData.case_id && (
                            <p><span className="font-semibold">Case ID:</span> {caseData.case_id}</p>
                          )}
                          {caseData.first_page && (
                            <p><span className="font-semibold">Pages:</span> {caseData.first_page} - {caseData.last_page}</p>
                          )}
                          {caseData.frontend_url && (
                            <a 
                              href={caseData.frontend_url} 
                              target="_blank" 
                              rel="noopener noreferrer"
                              className="text-legal-blue hover:underline flex items-center gap-1"
                            >
                              View on Case.Law <Link className="h-3 w-3" />
                            </a>
                          )}
                        </div>
                      </div>
                    </div>
                  </TabsContent>
                </CardContent>
              </Tabs>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Metadata */}
            <Card className="shadow-lg">
              <CardHeader>
                <CardTitle className="text-lg">Case Information</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <div>
                  <span className="font-semibold">Added to Database:</span>
                  <p className="text-muted-foreground">
                    {caseData.created_at ? new Date(caseData.created_at).toLocaleString() : 'Unknown'}
                  </p>
                </div>
                {caseData.court_info && (
                  <>
                    <Separator />
                    <div>
                      <span className="font-semibold">Court Information:</span>
                      <p className="text-muted-foreground">
                        {JSON.stringify(caseData.court_info)}
                      </p>
                    </div>
                  </>
                )}
                {caseData.volume && (
                  <>
                    <Separator />
                    <div>
                      <span className="font-semibold">Volume:</span>
                      <p className="text-muted-foreground">
                        {JSON.stringify(caseData.volume)}
                      </p>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CaseViewer;

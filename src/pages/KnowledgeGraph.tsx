import { useCallback } from "react";
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  MarkerType,
  BackgroundVariant,
} from "reactflow";
import "reactflow/dist/style.css";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, Network, Scale, User, FileText, BookOpen } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";
import { knowledgeGraphData } from "@/data/mockCases";

const getNodeColor = (type: string) => {
  switch (type) {
    case "case":
      return "#2563eb"; // legal-blue
    case "court":
      return "#f59e0b"; // legal-gold
    case "judge":
      return "#10b981"; // success
    case "statute":
      return "#8b5cf6"; // purple
    default:
      return "#6b7280";
  }
};

const getNodeIcon = (type: string) => {
  switch (type) {
    case "case":
      return "⚖️";
    case "court":
      return "🏛️";
    case "judge":
      return "👤";
    case "statute":
      return "📜";
    default:
      return "•";
  }
};

const KnowledgeGraph = () => {
  const navigate = useNavigate();

  const initialNodes: Node[] = knowledgeGraphData.nodes.map((node, idx) => ({
    id: node.id,
    type: "default",
    data: {
      label: (
        <div className="flex flex-col items-center gap-1 p-2">
          <div className="text-2xl">{getNodeIcon(node.type)}</div>
          <div className="text-xs font-semibold text-center">{node.label}</div>
          {node.level && <div className="text-[10px] text-gray-500">{node.level}</div>}
        </div>
      ),
    },
    position: {
      x: Math.cos((idx * 2 * Math.PI) / knowledgeGraphData.nodes.length) * 400 + 500,
      y: Math.sin((idx * 2 * Math.PI) / knowledgeGraphData.nodes.length) * 400 + 300,
    },
    style: {
      background: getNodeColor(node.type),
      color: "white",
      border: "2px solid white",
      borderRadius: "12px",
      padding: "10px",
      fontSize: "12px",
      fontWeight: 600,
      minWidth: "120px",
    },
  }));

  const getEdgeColor = (type: string) => {
    switch (type) {
      case "CONTRADICTS":
        return "#ef4444";
      case "SIMILAR_TO":
        return "#10b981";
      case "CITES":
        return "#8b5cf6";
      default:
        return "#6b7280";
    }
  };

  const initialEdges: Edge[] = knowledgeGraphData.edges.map((edge) => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    label: edge.type,
    type: "smoothstep",
    animated: edge.type === "CONTRADICTS" || edge.type === "SIMILAR_TO",
    style: {
      stroke: getEdgeColor(edge.type),
      strokeWidth: 2,
    },
    labelStyle: {
      fontSize: "10px",
      fontWeight: 600,
      fill: getEdgeColor(edge.type),
    },
    markerEnd: {
      type: MarkerType.ArrowClosed,
      color: getEdgeColor(edge.type),
    },
  }));

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    if (node.id.startsWith("case-")) {
      navigate(`/case/${node.id}`);
    }
  }, [navigate]);

  const stats = {
    totalNodes: nodes.length,
    cases: nodes.filter((n) => n.id.startsWith("case-")).length,
    courts: nodes.filter((n) => n.id.startsWith("court-")).length,
    relationships: edges.length,
    contradictions: edges.filter((e) => e.type === "CONTRADICTS").length,
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20">
      <div className="container mx-auto px-4 py-8 max-w-[98%]">
        <div className="mb-6">
          <Button variant="ghost" onClick={() => navigate("/")} className="mb-4">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Dashboard
          </Button>

          <div className="flex items-center gap-3 mb-4">
            <div className="h-12 w-12 rounded-lg bg-gradient-legal flex items-center justify-center">
              <Network className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold">Knowledge Graph</h1>
              <p className="text-muted-foreground">
                Neo4j-based legal case relationships and precedent chains
              </p>
            </div>
          </div>
        </div>

        {/* Stats Bar */}
        <div className="grid grid-cols-5 gap-4 mb-6">
          <Card>
            <CardContent className="pt-6">
              <div className="text-2xl font-bold">{stats.totalNodes}</div>
              <p className="text-xs text-muted-foreground">Total Nodes</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="text-2xl font-bold text-legal-blue">{stats.cases}</div>
              <p className="text-xs text-muted-foreground">Cases</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="text-2xl font-bold text-legal-gold">{stats.courts}</div>
              <p className="text-xs text-muted-foreground">Courts</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="text-2xl font-bold">{stats.relationships}</div>
              <p className="text-xs text-muted-foreground">Relationships</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="text-2xl font-bold text-destructive">{stats.contradictions}</div>
              <p className="text-xs text-muted-foreground">Contradictions</p>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Graph Visualization */}
          <Card className="lg:col-span-3 shadow-lg">
            <CardHeader>
              <CardTitle>Interactive Graph Visualization</CardTitle>
              <CardDescription>
                Click on case nodes to view details. Drag to explore relationships.
              </CardDescription>
            </CardHeader>
            <CardContent className="p-0">
              <div style={{ height: "700px" }} className="bg-muted/10 rounded-b-lg">
                <ReactFlow
                  nodes={nodes}
                  edges={edges}
                  onNodesChange={onNodesChange}
                  onEdgesChange={onEdgesChange}
                  onNodeClick={onNodeClick}
                  fitView
                  attributionPosition="bottom-left"
                >
                  <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
                  <Controls />
                </ReactFlow>
              </div>
            </CardContent>
          </Card>

          {/* Legend & Info */}
          <div className="space-y-6">
            <Card className="shadow-lg">
              <CardHeader>
                <CardTitle className="text-lg">Node Types</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center gap-3">
                  <div className="h-8 w-8 rounded-lg flex items-center justify-center text-lg" style={{ background: getNodeColor("case") }}>
                    ⚖️
                  </div>
                  <div>
                    <div className="font-semibold text-sm">Case</div>
                    <div className="text-xs text-muted-foreground">Legal decisions</div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="h-8 w-8 rounded-lg flex items-center justify-center text-lg" style={{ background: getNodeColor("court") }}>
                    🏛️
                  </div>
                  <div>
                    <div className="font-semibold text-sm">Court</div>
                    <div className="text-xs text-muted-foreground">Judicial bodies</div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="h-8 w-8 rounded-lg flex items-center justify-center text-lg" style={{ background: getNodeColor("judge") }}>
                    👤
                  </div>
                  <div>
                    <div className="font-semibold text-sm">Judge</div>
                    <div className="text-xs text-muted-foreground">Opinion authors</div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="h-8 w-8 rounded-lg flex items-center justify-center text-lg" style={{ background: getNodeColor("statute") }}>
                    📜
                  </div>
                  <div>
                    <div className="font-semibold text-sm">Statute</div>
                    <div className="text-xs text-muted-foreground">Legal codes</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="shadow-lg">
              <CardHeader>
                <CardTitle className="text-lg">Relationship Types</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <div className="flex items-center gap-2">
                  <div className="h-1 w-8 rounded" style={{ background: getEdgeColor("CITES") }} />
                  <span className="font-semibold">CITES</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-1 w-8 rounded" style={{ background: getEdgeColor("SIMILAR_TO") }} />
                  <span className="font-semibold">SIMILAR_TO</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-1 w-8 rounded" style={{ background: getEdgeColor("CONTRADICTS") }} />
                  <span className="font-semibold">CONTRADICTS</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-1 w-8 rounded bg-gray-500" />
                  <span className="font-semibold">DECIDED_BY</span>
                </div>
              </CardContent>
            </Card>

            <Card className="shadow-lg">
              <CardHeader>
                <CardTitle className="text-lg">Sample Cypher Query</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="bg-muted/50 p-3 rounded-lg font-mono text-xs overflow-x-auto">
                  <code className="text-success">
                    MATCH (c1:Case)-[:CONTRADICTS]-(c2:Case)
                    <br />
                    WHERE c1.category = 'Employment Law'
                    <br />
                    RETURN c1, c2
                  </code>
                </div>
                <p className="text-xs text-muted-foreground mt-2">
                  Find all contradictory employment law cases
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default KnowledgeGraph;

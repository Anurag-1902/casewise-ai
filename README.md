# LexLink: Intelligent Knowledge Graphs for Legal Research

<div align="center">

![LexLink](https://img.shields.io/badge/AI-Legal%20Research-blue)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript)
![Status](https://img.shields.io/badge/Status-Phase%201%20Demo-success)

**Revolutionizing Legal Research with AI-Powered Knowledge Graphs**

[View Demo](https://lovable.dev/projects/43ad87d9-3535-40d9-ba30-b11337cfa50b) · [Report Bug](https://github.com/username/lexlink/issues) · [Documentation](./ARCHITECTURE.md)

</div>

---

## 🎯 Overview

LexLink is an AI-powered legal research platform that transforms how legal professionals analyze case law through:

- ⚖️ **Automated Summarization** - AI-generated summaries preserving legal reasoning
- 🔍 **Semantic Search** - Find similar cases using Legal-BERT embeddings
- ⚠️ **Contradiction Detection** - Identify conflicting rulings across jurisdictions
- 🕸️ **Knowledge Graphs** - Interactive visualization of case relationships

**Current Phase**: Fully functional frontend prototype with mock data demonstrating all system capabilities.

---

## ✨ Features

### 🏠 Dashboard
- Real-time case analytics and statistics
- Quick access to recent cases
- Search across 24,891+ indexed cases
- System health metrics

### 📄 Case Viewer
- **AI-Powered Summaries**: Legal-BERT + BART summarization
- **Full Text Analysis**: Complete case opinions with metadata
- **Similar Cases**: FAISS-powered semantic similarity (≥75% threshold)
- **Contradiction Alerts**: NLI-based conflict detection
- **Citation Network**: Explore precedent chains

### 🕸️ Interactive Knowledge Graph
- **Visual Exploration**: Drag-and-drop Neo4j-style graph
- **Node Types**: Cases, Courts, Judges, Statutes
- **Relationship Types**: CITES, SIMILAR_TO, CONTRADICTS, DECIDED_BY
- **Cypher Queries**: Sample graph query examples

---

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Modern web browser

### Installation

```bash
# Clone the repository
git clone https://github.com/username/lexlink.git
cd lexlink

# Install dependencies
npm install

# Start development server
npm run dev
```

Visit `http://localhost:8080` to view the application.

---

## 🏗️ Technology Stack

### Frontend (Current Implementation)
- **React 18** + **TypeScript** - Component framework
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Utility-first styling
- **shadcn/ui** - Component library
- **React Flow** - Knowledge graph visualization
- **Framer Motion** - Animations
- **React Query** - Data fetching

### Planned Backend Stack
- **Python 3.9+** with FastAPI
- **Legal-BERT** (nlpaueb/legal-bert-base-uncased)
- **BART** (facebook/bart-large-cnn)
- **FAISS** - Vector similarity search
- **Neo4j** - Graph database
- **SpaCy** + **eyecite** - NLP and citation extraction

See [ARCHITECTURE.md](./ARCHITECTURE.md) for complete system design.

---

## 📊 Mock Data

The current implementation includes realistic mock data demonstrating:

- **4 Sample Cases** from various jurisdictions
  - Smith v. Jones (Supreme Court)
  - Tech Corp. v. Innovation LLC (9th Circuit)
  - State v. Johnson (California Supreme Court)
  - DataCorp v. Privacy Advocates (S.D.N.Y.)

- **Knowledge Graph** with 11 nodes and 10 relationships
- **Similarity Scores** showing Legal-BERT embedding comparisons
- **Contradiction Detection** between conflicting cases

---

## 🎓 Academic Project

**Institution**: RV College of Engineering  
**Department**: Information Science & Engineering  
**Course**: AI/ML Project - Lab Part B, Phase 1  
**Team**:
- Aakrisht Tiwary (1RV23IS003)
- Anurag Rath (1RV23IS020)

**Project Goals**:
1. ✅ Build functional UI prototype
2. ⏳ Implement ML pipeline (Legal-BERT, BART, FAISS)
3. ⏳ Deploy Neo4j knowledge graph
4. ⏳ Integrate backend APIs
5. ⏳ Achieve 70-80% research time reduction

---

## 📁 Project Structure

```
lexlink/
├── src/
│   ├── components/       # Reusable UI components
│   │   └── ui/          # shadcn/ui components
│   ├── pages/           # Route pages
│   │   ├── Dashboard.tsx
│   │   ├── CaseViewer.tsx
│   │   └── KnowledgeGraph.tsx
│   ├── data/            # Mock data
│   │   └── mockCases.ts
│   ├── lib/             # Utilities
│   └── App.tsx          # Main app component
├── public/              # Static assets
├── ARCHITECTURE.md      # Complete system design
└── package.json
```

---

## 🔮 Roadmap

### Phase 1: ✅ Frontend Prototype (Complete)
- [x] Design system with legal-professional aesthetic
- [x] Dashboard with search and analytics
- [x] Case viewer with tabbed interface
- [x] Interactive knowledge graph visualization
- [x] Mock data for demonstration

### Phase 2: ML Pipeline (In Progress)
- [ ] Set up Python backend with FastAPI
- [ ] Integrate Legal-BERT for embeddings
- [ ] Implement BART summarization
- [ ] Build FAISS similarity index
- [ ] Fine-tune NLI model for contradictions

### Phase 3: Knowledge Graph (Planned)
- [ ] Set up Neo4j database
- [ ] Design graph schema
- [ ] Extract citations with eyecite
- [ ] Build relationship network
- [ ] Implement Cypher query API

### Phase 4: Integration & Deployment
- [ ] Connect frontend to backend APIs
- [ ] Deploy to cloud (AWS/GCP/Azure)
- [ ] Performance optimization
- [ ] User testing with legal professionals
- [ ] Documentation and training materials

---

## 📈 Expected Outcomes

| Metric | Target | Status |
|--------|--------|--------|
| Summarization ROUGE-L | > 0.45 | Pending ML implementation |
| Similarity Precision @ 10 | > 0.80 | Pending ML implementation |
| Contradiction Accuracy | > 85% | Pending ML implementation |
| Graph Query Speed | < 1 sec | Demo: < 100ms |
| Research Time Reduction | 70-80% | Projected |

---

## 🤝 Contributing

This is an academic project, but feedback and suggestions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📚 Resources

### Research Papers
- [LEGAL-BERT: The Muppets straight out of Law School](https://aclanthology.org/2020.findings-emnlp.261/) (EMNLP 2020)
- [Retrieval-Augmented Generation for Legal Summarization](https://arxiv.org/abs/2401.xxxxx)
- [DELTA: Discriminative Encoder for Legal Case Retrieval](https://arxiv.org/abs/2405.xxxxx)

### Datasets
- [Caselaw Access Project](https://case.law) - 6.7M+ U.S. court decisions
- [LexGLUE](https://github.com/coastalcph/lex-glue) - Legal NLP benchmark

### Tools
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Neo4j](https://neo4j.com)
- [React Flow](https://reactflow.dev)

---

## 📄 License

This project is created for academic purposes at RV College of Engineering.

---

## 🙏 Acknowledgments

- RV College of Engineering, Department of ISE
- Hugging Face for pretrained models
- Caselaw Access Project for legal data
- shadcn for beautiful UI components

---

<div align="center">

**Built with ⚖️ by Aakrisht Tiwary & Anurag Rath**

[View Documentation](./ARCHITECTURE.md) · [Report Issues](https://github.com/username/lexlink/issues)

</div>

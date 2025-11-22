# LexLink: Intelligent Knowledge Graphs for Legal Research
## Complete System Architecture & Implementation Guide

---

## 🎯 Project Overview

LexLink is an AI-powered legal research system that revolutionizes how legal professionals analyze case law by providing:

1. **Automated Document Summarization** - Using Legal-BERT and BART models
2. **Semantic Similarity Search** - FAISS-powered vector search for finding related cases
3. **Contradiction Detection** - NLI-based identification of conflicting rulings
4. **Interactive Knowledge Graphs** - Neo4j visualization of case relationships

**Current Implementation**: This is a fully functional **frontend prototype** built with React + TypeScript that demonstrates all system capabilities with mock data and interactive visualizations.

---

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE LAYER                     │
│  (React + TypeScript + Tailwind CSS + shadcn/ui)           │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │  Dashboard   │  │ Case Viewer  │  │ Knowledge Graph │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      API/SERVICE LAYER                       │
│                  (To be implemented)                         │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ Summarization│  │  Similarity  │  │  Contradiction  │  │
│  │   Service    │  │   Service    │  │    Service      │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                         AI LAYER                             │
│                  (Python Backend Required)                   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Legal-BERT (nlpaueb/legal-bert-base-uncased)        │  │
│  │  • 768-dimensional embeddings                        │  │
│  │  • Domain-specific legal language understanding      │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ BART (facebook/bart-large-cnn)                       │  │
│  │  • Abstractive summarization                         │  │
│  │  • Preserves legal reasoning and citations          │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ FAISS (Facebook AI Similarity Search)               │  │
│  │  • IndexFlatL2 for L2 distance                      │  │
│  │  • k-NN search with k=100                           │  │
│  │  • Cosine similarity threshold ≥ 0.75               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      DATA LAYER                              │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Neo4j Knowledge Graph                                │  │
│  │  • Nodes: Case, Court, Judge, Statute               │  │
│  │  • Edges: CITES, SIMILAR_TO, CONTRADICTS, etc.     │  │
│  │  • Cypher query language                            │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Case Law Database                                    │  │
│  │  • Source: Caselaw Access Project (case.law)       │  │
│  │  • 6.7M+ U.S. court decisions                       │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Module Implementation Details

### 1. Document Summarization Module

**Objective**: Generate concise, accurate summaries preserving legal reasoning, holdings, and citations.

**Implementation**:

```python
# Stage 1: Extractive Summarization (Legal-BERT + TextRank)
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

class LegalSummarizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        self.model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
    
    def get_sentence_embeddings(self, sentences):
        """Generate Legal-BERT embeddings for sentences"""
        embeddings = []
        for sentence in sentences:
            inputs = self.tokenizer(sentence, return_tensors="pt", 
                                   max_length=512, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(embedding[0])
        return np.array(embeddings)
    
    def textrank_summarize(self, text, ratio=0.3):
        """Extract key sentences using TextRank algorithm"""
        # Split into sentences
        sentences = self._split_sentences(text)
        
        # Generate embeddings
        embeddings = self.get_sentence_embeddings(sentences)
        
        # Build similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Create graph and apply PageRank
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        
        # Select top sentences
        ranked_sentences = sorted(
            ((scores[i], s) for i, s in enumerate(sentences)), 
            reverse=True
        )
        
        summary_length = int(len(sentences) * ratio)
        summary_sentences = [s for _, s in ranked_sentences[:summary_length]]
        
        return ' '.join(summary_sentences)

# Stage 2: Abstractive Refinement (BART)
from transformers import BartForConditionalGeneration, BartTokenizer

class AbstractiveSummarizer:
    def __init__(self):
        self.model = BartForConditionalGeneration.from_pretrained(
            "facebook/bart-large-cnn"
        )
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    
    def summarize(self, text, max_length=250):
        """Generate abstractive summary"""
        inputs = self.tokenizer([text], max_length=1024, 
                               return_tensors="pt", truncation=True)
        
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=100,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        summary = self.tokenizer.decode(summary_ids[0], 
                                       skip_special_tokens=True)
        return summary
```

**Expected Performance**:
- ROUGE-L Score: > 0.45
- Preserves 90%+ of critical legal holdings
- Processing time: ~5-10 seconds per case

---

### 2. Semantic Similarity Search Module

**Objective**: Find semantically similar cases using vector embeddings and FAISS.

**Implementation**:

```python
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel

class SimilaritySearchEngine:
    def __init__(self, dimension=768):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance
        self.tokenizer = AutoTokenizer.from_pretrained(
            "nlpaueb/legal-bert-base-uncased"
        )
        self.model = AutoModel.from_pretrained(
            "nlpaueb/legal-bert-base-uncased"
        )
        self.case_ids = []
    
    def generate_embedding(self, text):
        """Generate 768-d Legal-BERT embedding"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.numpy()[0]
    
    def index_case(self, case_id, case_text):
        """Add case to FAISS index"""
        embedding = self.generate_embedding(case_text)
        self.index.add(np.array([embedding]).astype('float32'))
        self.case_ids.append(case_id)
    
    def search_similar(self, query_text, k=100, threshold=0.75):
        """Find k most similar cases"""
        query_embedding = self.generate_embedding(query_text)
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, k)
        
        # Convert L2 distance to cosine similarity
        # similarity = 1 - (distance / 2)  # Approximation
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            similarity = 1 - (dist / 2)
            if similarity >= threshold:
                results.append({
                    'case_id': self.case_ids[idx],
                    'similarity': float(similarity),
                    'distance': float(dist)
                })
        
        return results
```

**Expected Performance**:
- Precision @ k=10: > 0.80
- Query time: < 100ms for 1M cases
- Recall: > 0.75 for semantically similar cases

---

### 3. Contradiction Detection Module

**Objective**: Identify conflicting rulings using NLI (Natural Language Inference).

**Implementation**:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class ContradictionDetector:
    def __init__(self):
        # Fine-tuned Legal-BERT on NLI task
        self.tokenizer = AutoTokenizer.from_pretrained(
            "nlpaueb/legal-bert-base-uncased"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "nlpaueb/legal-bert-base-uncased",
            num_labels=3  # entailment, contradiction, neutral
        )
        # Note: Requires fine-tuning on legal NLI dataset
    
    def detect_contradiction(self, case1_text, case2_text):
        """
        Classify relationship between two cases
        Returns: 'consistent', 'contradictory', or 'neutral'
        """
        inputs = self.tokenizer(
            case1_text,
            case2_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        
        labels = ['consistent', 'contradictory', 'neutral']
        confidence = torch.softmax(logits, dim=1)[0][prediction].item()
        
        return {
            'relationship': labels[prediction],
            'confidence': float(confidence)
        }
    
    def categorize_conflict(self, case1, case2):
        """Categorize type of contradiction"""
        # Temporal conflict: later case overrules earlier
        if case1['date'] < case2['date']:
            return 'temporal_conflict'
        
        # Jurisdictional divergence: different courts, same issue
        if case1['court'] != case2['court']:
            return 'jurisdictional_divergence'
        
        # Reasoning reversal: same court, opposite holding
        return 'reasoning_reversal'
```

**Expected Performance**:
- Accuracy: > 85%
- False Positive Rate: < 15%
- Precision: > 0.82
- Recall: > 0.78

---

### 4. Knowledge Graph Construction Module

**Objective**: Build Neo4j graph connecting cases, courts, judges, and statutes.

**Implementation**:

```python
from neo4j import GraphDatabase
import spacy
import eyecite

class KnowledgeGraphBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.nlp = spacy.load("en_core_web_sm")
        
    def create_case_node(self, case_data):
        """Create Case node in Neo4j"""
        with self.driver.session() as session:
            query = """
            CREATE (c:Case {
                id: $id,
                title: $title,
                citation: $citation,
                court: $court,
                date: $date,
                jurisdiction: $jurisdiction,
                summary: $summary,
                embedding: $embedding
            })
            RETURN c
            """
            session.run(query, **case_data)
    
    def create_citation_relationship(self, case_id, cited_case_id):
        """Create CITES relationship"""
        with self.driver.session() as session:
            query = """
            MATCH (c1:Case {id: $case_id})
            MATCH (c2:Case {id: $cited_case_id})
            CREATE (c1)-[:CITES]->(c2)
            """
            session.run(query, case_id=case_id, 
                       cited_case_id=cited_case_id)
    
    def create_similarity_relationship(self, case1_id, case2_id, 
                                      similarity_score):
        """Create SIMILAR_TO relationship"""
        with self.driver.session() as session:
            query = """
            MATCH (c1:Case {id: $case1_id})
            MATCH (c2:Case {id: $case2_id})
            CREATE (c1)-[:SIMILAR_TO {score: $score}]->(c2)
            """
            session.run(query, case1_id=case1_id, 
                       case2_id=case2_id, score=similarity_score)
    
    def create_contradiction_relationship(self, case1_id, case2_id, 
                                         conflict_type):
        """Create CONTRADICTS relationship"""
        with self.driver.session() as session:
            query = """
            MATCH (c1:Case {id: $case1_id})
            MATCH (c2:Case {id: $case2_id})
            CREATE (c1)-[:CONTRADICTS {type: $conflict_type}]->(c2)
            """
            session.run(query, case1_id=case1_id, 
                       case2_id=case2_id, conflict_type=conflict_type)
    
    def query_precedent_chain(self, case_id):
        """Find precedent chain using shortest path"""
        with self.driver.session() as session:
            query = """
            MATCH path = shortestPath(
                (start:Case {id: $case_id})-[:CITES*]->
                (end:Case)
            )
            RETURN path
            """
            result = session.run(query, case_id=case_id)
            return [record["path"] for record in result]
    
    def query_contradictory_cases(self, category):
        """Find all contradictory cases in a category"""
        with self.driver.session() as session:
            query = """
            MATCH (c1:Case)-[r:CONTRADICTS]-(c2:Case)
            WHERE c1.category = $category
            RETURN c1, c2, r
            """
            result = session.run(query, category=category)
            return list(result)
```

**Sample Cypher Queries**:

```cypher
// Find all cases with contradictory rulings in employment law
MATCH (c1:Case)-[:CONTRADICTS]-(c2:Case)
WHERE c1.category = 'Employment Law'
RETURN c1.title, c2.title, c1.date, c2.date

// Find precedent chain for a specific case
MATCH path = shortestPath(
    (start:Case {id: 'case-001'})-[:CITES*]->(end:Case)
)
RETURN path

// Find most influential judges (by cases authored)
MATCH (j:Judge)-[:AUTHORED]->(c:Case)
RETURN j.name, count(c) as cases_authored
ORDER BY cases_authored DESC
LIMIT 10

// Find cases with high similarity but no direct citations
MATCH (c1:Case)-[s:SIMILAR_TO]-(c2:Case)
WHERE s.score > 0.85 
AND NOT (c1)-[:CITES]-(c2)
RETURN c1.title, c2.title, s.score
```

---

## 🚀 Implementation Roadmap

### Phase 1: Data Collection & Preprocessing (Week 1-2)
- [ ] Set up Caselaw Access Project API access
- [ ] Download and preprocess case dataset (focus on specific domains)
- [ ] Extract metadata, citations, and full text
- [ ] Clean and normalize data
- [ ] Set up PostgreSQL database for raw case storage

### Phase 2: Model Training & Integration (Week 3-5)
- [ ] Set up Python ML environment (PyTorch, Transformers, FAISS)
- [ ] Load Legal-BERT and BART pretrained models
- [ ] Fine-tune BART on legal summarization task
- [ ] Fine-tune Legal-BERT for NLI (contradiction detection)
- [ ] Generate embeddings for all cases
- [ ] Build and test FAISS index

### Phase 3: Knowledge Graph Construction (Week 6-7)
- [ ] Set up Neo4j database
- [ ] Design graph schema (nodes, relationships)
- [ ] Extract citations using eyecite
- [ ] Extract entities using SpaCy
- [ ] Populate graph with cases, courts, judges, statutes
- [ ] Create relationships (CITES, SIMILAR_TO, CONTRADICTS)
- [ ] Optimize graph queries

### Phase 4: API Development (Week 8-9)
- [ ] Build Flask/FastAPI backend
- [ ] Implement summarization endpoint
- [ ] Implement similarity search endpoint
- [ ] Implement contradiction detection endpoint
- [ ] Implement knowledge graph query endpoint
- [ ] Add authentication and rate limiting
- [ ] Write API documentation (OpenAPI/Swagger)

### Phase 5: Testing & Evaluation (Week 10-11)
- [ ] Evaluate summarization (ROUGE, BLEU scores)
- [ ] Evaluate similarity search (precision, recall)
- [ ] Evaluate contradiction detection (accuracy, F1)
- [ ] Benchmark query performance
- [ ] Conduct user testing with legal professionals
- [ ] Iterate based on feedback

### Phase 6: Deployment & Documentation (Week 12)
- [ ] Deploy backend (AWS/GCP/Azure)
- [ ] Set up CI/CD pipeline
- [ ] Configure monitoring and logging
- [ ] Write comprehensive documentation
- [ ] Create video tutorials
- [ ] Prepare research paper / presentation

---

## 📁 Recommended Directory Structure

```
lexlink/
├── frontend/                 # React frontend (current implementation)
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── data/
│   │   └── lib/
│   └── package.json
│
├── backend/                  # Python backend (to be implemented)
│   ├── app/
│   │   ├── api/
│   │   │   ├── summarization.py
│   │   │   ├── similarity.py
│   │   │   ├── contradiction.py
│   │   │   └── graph.py
│   │   ├── models/
│   │   │   ├── legal_bert.py
│   │   │   ├── bart.py
│   │   │   └── nli.py
│   │   ├── services/
│   │   │   ├── faiss_service.py
│   │   │   ├── neo4j_service.py
│   │   │   └── preprocessing.py
│   │   └── utils/
│   ├── tests/
│   ├── requirements.txt
│   └── main.py
│
├── data/                     # Data storage
│   ├── raw/
│   ├── processed/
│   ├── embeddings/
│   └── models/
│
├── notebooks/               # Jupyter notebooks for experiments
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
│
├── scripts/                 # Utility scripts
│   ├── download_data.py
│   ├── preprocess.py
│   ├── generate_embeddings.py
│   └── build_graph.py
│
└── docs/
    ├── API.md
    ├── DEPLOYMENT.md
    └── RESEARCH.md
```

---

## 🛠️ Technology Stack

### Frontend (Current)
- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **UI Components**: shadcn/ui
- **Visualization**: React Flow (knowledge graph)
- **Animation**: Framer Motion
- **State Management**: React Query

### Backend (To Implement)
- **Language**: Python 3.9+
- **Web Framework**: FastAPI
- **ML Framework**: PyTorch, Hugging Face Transformers
- **Vector Search**: FAISS
- **Graph Database**: Neo4j
- **NLP**: SpaCy, eyecite
- **Data Processing**: Pandas, NumPy

### Infrastructure (To Implement)
- **Cloud Platform**: AWS / GCP / Azure
- **Container**: Docker
- **Orchestration**: Kubernetes (optional)
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus + Grafana

---

## 📈 Expected Outcomes & KPIs

### Performance Metrics
1. **Summarization**:
   - ROUGE-L > 0.45
   - BLEU > 0.40
   - Human evaluation: 4.0+/5.0

2. **Similarity Search**:
   - Precision @ k=10: > 0.80
   - Recall @ k=10: > 0.75
   - Query latency: < 100ms

3. **Contradiction Detection**:
   - Accuracy: > 85%
   - Precision: > 0.82
   - Recall: > 0.78
   - F1 Score: > 0.80

4. **System Performance**:
   - Knowledge graph queries: < 1 second
   - Research time reduction: 70-80%
   - User satisfaction: 4.5+/5.0

---

## 🔒 Security & Privacy Considerations

- **Data Encryption**: All data encrypted at rest and in transit
- **Access Control**: Role-based authentication (RBAC)
- **Audit Logging**: Complete audit trail for all operations
- **GDPR Compliance**: User data privacy and right to deletion
- **API Security**: Rate limiting, API keys, OAuth 2.0

---

## 📚 References & Resources

### Key Papers
1. Chalkidis et al., "LEGAL-BERT: The Muppets straight out of Law School" (EMNLP 2020)
2. Zhong et al., "JEC-QA: A Legal-Domain Question Answering Dataset" (AAAI 2020)
3. Mentzingen et al., "Textual Similarity in Legal Precedent Retrieval" (2020)

### Datasets
- Caselaw Access Project: https://case.law
- EUR-Lex: https://eur-lex.europa.eu
- LexGLUE: Legal benchmark dataset

### Tools & Libraries
- Hugging Face Transformers: https://huggingface.co/docs/transformers
- FAISS: https://github.com/facebookresearch/faiss
- Neo4j: https://neo4j.com/docs
- SpaCy: https://spacy.io
- eyecite: https://github.com/freelawproject/eyecite

---

## 🎓 Academic Integration

This project fulfills the requirements for:
- **Course**: Artificial Intelligence and Machine Learning - Project Lab Part B
- **Institution**: RV College of Engineering
- **Department**: Information Science & Engineering
- **Students**: Aakrisht Tiwary (1RV23IS003), Anurag Rath (1RV23IS020)

**Faculty Advisor**: [Add advisor name]
**Submission Date**: Phase 1 - November 2024

---

## 📞 Support & Contact

For questions or collaboration:
- GitHub: [Add repository link]
- Email: [Add contact email]
- Documentation: [Add docs link]

---

**Note**: This document provides the complete architecture for LexLink. The current implementation is a fully functional frontend prototype. The Python ML backend requires separate implementation following the specifications provided above.

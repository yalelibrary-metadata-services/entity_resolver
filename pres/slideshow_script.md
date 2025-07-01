# From Words to Vectors: Text Embeddings and Entity Resolution
## Yale AI Workshop - Complete Slideshow Script

**Duration**: 45 minutes  
**Format**: 30 slides + 3 coding sessions  
**Audience**: Yale graduate students (STEM to humanities)

---

## INTRODUCTION & SETUP (4 minutes, 3 slides)

### Slide 1: Welcome & Overview (90 seconds)
**Visual**: workshop_overview.png - Learning objectives and journey map

**Script**:
"Welcome to 'From Words to Vectors' - a journey through one of the most fascinating challenges in AI: how do we teach computers to understand when two pieces of text refer to the same person? Today, we'll follow the actual journey we took building Yale University Library's entity resolution system, from simple text embeddings to a production system processing 17.6 million catalog records.

We'll learn through three hands-on notebooks that tell this story chronologically - including our failures, breakthroughs, and the surprising insights that emerged along the way."

**Timing**: 90 seconds  
**Transition**: "Let me start with a concrete problem that will thread through our entire workshop..."

---

### Slide 2: The Franz Schubert Problem (90 seconds)
**Visual**: franz_schubert_decision_tree.png - Two different Franz Schuberts

**Script**:
"Meet our protagonist: Franz Schubert. Not the famous composer from 1797-1828, but a different Franz Schubert - a photographer born in 1930. Our Yale catalog contains records for both, and this seemingly simple case became the driving force behind our entire system.

Look at these records: 'Schubert, Franz - Symphony No. 9' versus 'Schubert, Franz - Archaeological Photography Methods.' Same name, completely different people, different centuries. How do we teach a computer to distinguish between them? This question will guide everything we build today."

**Timing**: 90 seconds  
**Transition**: "This problem led us on a journey through three major phases..."

---

### Slide 3: The Journey Ahead (60 seconds)
**Visual**: journey_timeline.png - Evolution from embeddings to production

**Script**:
"Our journey unfolds in three acts: First, we'll explore text embeddings and discover why simple similarity isn't enough. Second, we'll add domain classification to provide context. Finally, we'll integrate everything into a production system with vector databases and hot-deck imputation.

Each section combines conceptual slides with hands-on coding in our notebooks. By the end, you'll understand both the theory and practice of building large-scale entity resolution systems."

**Timing**: 60 seconds  
**Transition**: "Let's begin with the foundation: how text becomes numbers..."

---

## PART 1: EMBEDDINGS FUNDAMENTALS (15 minutes, 8 slides + coding)

### Slide 4: The Evolution of Text Embeddings (60 seconds)
**Visual**: embedding_evolution_chart.png - Word2Vec to modern transformers

**Script**:
"Text embeddings have evolved dramatically. We started with Word2Vec in 2013 - revolutionary for its time. But look at this progression: GloVe, FastText, BERT, GPT-3, and now OpenAI's text-embedding-3-small. Each breakthrough solved limitations of the previous generation.

For our Yale project, we chose text-embedding-3-small: 1,536 dimensions, excellent semantic understanding, cost-effective API, and crucially - designed specifically for similarity tasks like entity resolution."

**Timing**: 60 seconds  
**Transition**: "But before text becomes vectors, it must become tokens..."

---

### Slide 5: Tokenization - The Hidden Foundation (60 seconds)
**Visual**: tokenization_comparison.png - Multi-language token analysis

**Script**:
"Tokenization is where language meets computation. Watch what happens when we analyze the same concept across languages: 'The influence of baroque music on contemporary literature' becomes different numbers of tokens in English, Spanish, Portuguese, and Chinese.

This matters enormously for cost and performance. Our Yale catalog includes records in many languages, and tokenization efficiency varies dramatically. English averages 3.5 characters per token, but Chinese can be much less efficient. This knowledge shaped our entire cost model."

**Timing**: 60 seconds  
**Transition**: "Let's see tokenization in action with our notebooks..."

---

### 🧪 **CODING BREAK #1: Tokenization & Similarity (7 minutes)**
**Notebook**: yale_embeddings_fundamentals.ipynb  
**Focus**: Cells 8-16 (tokenization analysis and similarity calculations)

**Instructions for students**:
1. **Run tokenization analysis** (2 min): Execute cells 8-10 to see how different texts tokenize
2. **Calculate embeddings** (2 min): Run cells 12-14 to create embeddings for Franz Schubert examples  
3. **Test similarity** (3 min): Run cells 14-16 to calculate cosine similarity between different text pairs

**Key observations to highlight**:
- Token counts vary significantly across languages and text types
- High similarity between name variations (0.95+)
- Moderate similarity between different Franz Schuberts (0.7-0.8)
- This moderate similarity is the core problem we need to solve

---

### Slide 6: Semantic Similarity in Action (60 seconds)
**Visual**: similarity_heatmap.png - Similarity matrix of catalog examples

**Script**:
"Here's what we discovered in your notebooks: embeddings excel at capturing semantic similarity. 'Schubert, Franz' and 'Franz Schubert' score 0.98 similarity - nearly perfect. Even across languages, 'Photography in archaeology' and 'Fotografía en arqueología' score 0.89.

But notice the challenge: our two Franz Schuberts still score 0.76 similarity. That's high enough to suggest they're related, but they're completely different people. A simple threshold can't solve this."

**Timing**: 60 seconds  
**Transition**: "This led us to a fundamental insight..."

---

### Slide 7: The Threshold Problem (60 seconds)
**Visual**: threshold_problem_demo.png - Why single thresholds fail

**Script**:
"We tested every threshold from 0.5 to 0.95. The results were sobering: no single threshold works for all entity types. Common names like 'Smith, John' require high thresholds, but unique names like 'Dostoyevsky' work with lower thresholds.

Our Franz Schubert case exemplifies this perfectly - the composer and photographer score 0.76 similarity. Set the threshold below 0.76, and you get false positives. Set it above, and you miss legitimate matches for other entities."

**Timing**: 60 seconds  
**Transition**: "The threshold problem forced us to think beyond similarity..."

---

### Slide 8: Cost Analysis - Real-World Constraints (60 seconds)
**Visual**: cost_benefit_analysis.png - Scaling considerations

**Script**:
"Before moving forward, we needed to understand the economics. Yale's 17.6 million records would cost $52,800 to embed with OpenAI's standard pricing - but only $26,400 with batch processing.

This cost analysis influenced every architectural decision. We couldn't afford to embed multiple variations or iterate carelessly. Every API call needed to count. This economic reality shapes how production AI systems are actually built."

**Timing**: 60 seconds  
**Transition**: "With these constraints in mind, we needed additional features..."

---

## PART 2: DOMAIN CLASSIFICATION (15 minutes, 9 slides + coding)

### Slide 9: Why Domain Context Matters (60 seconds)  
**Visual**: domain_taxonomy_tree.png - Yale classification hierarchy

**Script**:
"Our breakthrough insight: Franz Schubert the composer works in 'Music, Sound, and Sonic Arts' while Franz Schubert the photographer works in 'Documentary and Technical Arts.' Domain classification provides the missing context that pure text similarity cannot capture.

We developed a comprehensive taxonomy covering human knowledge domains: Arts, Sciences, Humanities, and Society. Each with detailed subcategories. This taxonomy became the foundation for disambiguating ambiguous entities."

**Timing**: 60 seconds  
**Transition**: "We needed a classification approach that could handle this taxonomy..."

---

### Slide 10: SetFit - The Initial Choice (45 seconds)
**Visual**: setfit_vs_mistral_matrix.png - Capability comparison

**Script**:
"SetFit seemed perfect: few-shot learning with just 2-3 examples per class, built on sentence transformers, fast training. We were excited to implement it for our domain classification needs.

But we quickly hit a wall: SetFit's underlying model has a 128-token limit. Our realistic catalog records often exceed 200 tokens. We faced a fundamental choice: truncate our data and lose context, or find an alternative."

**Timing**: 45 seconds  
**Transition**: "Let's examine this limitation in detail..."

---

### Slide 11: The Token Length Discovery (75 seconds)
**Visual**: tokenization_comparison.png - Real records vs. token limits

**Script**:
"Here's what broke our SetFit approach: real catalog records are long. A typical record includes contributor information, full titles, attribution details, subject classifications, and publication information. In multiple languages.

Look at this analysis: while simple names fit easily within 128 tokens, realistic catalog records range from 150-300 tokens. SetFit could handle maybe 25% of our actual data. The rest would be truncated, losing crucial context for classification."

**Timing**: 75 seconds  
**Transition**: "This forced us to explore alternatives..."

---

### Slide 12: Mistral Classifier Factory - The Solution (60 seconds)
**Visual**: setfit_vs_mistral_matrix.png - Detailed comparison

**Script**:
"Mistral Classifier Factory solved our problems elegantly: 32,000 token context length, excellent few-shot learning, strong multilingual support, and no training required. The cost difference was minimal - about $0.001 per classification - but the capability difference was transformative.

Most importantly, Mistral could process our complete catalog records without truncation, preserving all the context needed for accurate domain classification."

**Timing**: 60 seconds  
**Transition**: "Let's test both approaches in our notebooks..."

---

### 🧪 **CODING BREAK #2: SetFit vs Mistral (7 minutes)**
**Notebook**: yale_domain_classification.ipynb  
**Focus**: Cells 14-20 (comparing approaches with real data)

**Instructions for students**:
1. **Test SetFit limitations** (2 min): Run cells 14-16 to see token length problems
2. **Try Mistral classification** (3 min): Execute cells 18-20 to test domain classification
3. **Compare results** (2 min): Analyze which approach handles full catalog records better

**Key observations to highlight**:
- SetFit fails on realistic catalog records due to token limits
- Mistral handles full records without truncation
- Domain classification successfully distinguishes different Franz Schuberts
- This breakthrough enabled our production system

---

### Slide 13: Taxonomy in Action (60 seconds)
**Visual**: domain_taxonomy_tree.png - Classification results

**Script**:
"Domain classification transformed our Franz Schubert problem. The composer's records consistently classify as 'Music, Sound, and Sonic Arts' while the photographer's records classify as 'Documentary and Technical Arts.'

This gives us a powerful new feature: domain dissimilarity. When two entities have different domains, they're likely different people. When domains match, we examine other features more carefully."

**Timing**: 60 seconds  
**Transition**: "The combination of embeddings and domain classification was powerful, but we needed more..."

---

### Slide 14: Performance Analysis (45 seconds)
**Visual**: setfit_vs_mistral_matrix.png - Detailed metrics

**Script**:
"Our analysis showed Mistral's clear advantages: handling 100% of our data versus SetFit's 75%, processing complete context versus truncated records, and providing consistent performance across all entity types.

The cost was reasonable - $18,000 to classify all 17.6 million records versus $1.76 million for manual classification. The ROI was compelling."

**Timing**: 45 seconds  
**Transition**: "With domain classification solved, we needed to integrate everything..."

---

### Slide 15: Integration Strategy (45 seconds)
**Visual**: journey_timeline.png - Progress through phases

**Script**:
"We now had two powerful tools: semantic similarity from embeddings and contextual understanding from domain classification. But production systems require more: vector databases for scale, missing data imputation, feature engineering, and robust classification.

Our next phase would combine these elements into a complete entity resolution pipeline."

**Timing**: 45 seconds  
**Transition**: "This brings us to our final phase: production integration..."

---

## PART 3: COMPLETE PIPELINE (18 minutes, 10 slides + coding)

### Slide 16: Vector Databases - Scaling Similarity Search (60 seconds)
**Visual**: weaviate_integration.png - Vector database workflow

**Script**:
"Comparing 17.6 million records requires 155 trillion pairwise comparisons - computationally impossible. Vector databases like Weaviate solve this with approximate nearest neighbor search, reducing comparisons by 99.23% while maintaining accuracy.

Weaviate integrates directly with OpenAI, automatically embedding and indexing our data. It provides production-scale similarity search with millisecond response times, enabling real-time entity resolution."

**Timing**: 60 seconds  
**Transition**: "But production systems need clean data..."

---

### Slide 17: Vector Hot-Deck Imputation (75 seconds)
**Visual**: hotdeck_imputation_flow.png - Missing data imputation process

**Script**:
"Traditional hot-deck imputation uses statistical similarity to fill missing values. Vector hot-deck imputation uses semantic similarity - a revolutionary approach.

When a record has missing subject classifications, we find semantically similar records using embeddings and copy their subject values. This isn't random - it's intelligent, context-aware data enhancement that improves both data quality and classification accuracy."

**Timing**: 75 seconds  
**Transition**: "Let's see this in action..."

---

### Slide 18: Feature Engineering for Entity Resolution (60 seconds)
**Visual**: feature_importance_radar.png - ML feature weights

**Script**:
"Production entity resolution combines multiple features: person name similarity, full record similarity, person-title interactions, domain dissimilarity, and birth-death matching. Each captures different aspects of entity identity.

Our logistic regression classifier learns optimal weights for each feature. Notice how domain dissimilarity has high negative weight - different domains strongly indicate different people, solving our Franz Schubert problem."

**Timing**: 60 seconds  
**Transition**: "Now let's experience the complete pipeline..."

---

### 🧪 **CODING BREAK #3: Complete Pipeline Demo (8 minutes)**
**Notebook**: yale_entity_resolution_pipeline.ipynb  
**Focus**: Cells 12-24 (complete pipeline with Franz Schubert disambiguation)

**Instructions for students**:
1. **Vector similarity search** (2 min): Run cells 10-12 to find similar records
2. **Hot-deck imputation** (3 min): Execute cells 12-14 to fill missing subject fields  
3. **Feature engineering** (2 min): Run cells 14-16 to calculate classification features
4. **Franz Schubert test** (1 min): Execute cells 19-20 to see disambiguation in action

**Key observations to highlight**:
- Vector search efficiently finds similar records
- Hot-deck imputation intelligently fills missing data
- Multiple features provide robust classification
- Franz Schubert composer vs photographer successfully distinguished

---

### Slide 19: Production Performance Metrics (75 seconds)
**Visual**: production_metrics_dashboard.png - Yale system results

**Script**:
"Our production system achieves remarkable results: 99.75% precision with 82.48% recall on 14,930 test pairs. This means only 25 false positives out of 10,000 predicted matches - extraordinary accuracy for such a challenging problem.

The system processes Yale's complete 17.6 million record catalog, identifying entity clusters with confidence scores. Human reviewers only need to examine the 25 false positives plus 2,111 missed matches - a 99.23% reduction in manual work."

**Timing**: 75 seconds  
**Transition**: "Let's examine the broader impact..."

---

### Slide 20: Computational Efficiency (60 seconds)
**Visual**: pipeline_architecture.png - Complete system architecture

**Script**:
"The complete system demonstrates the power of modern AI architecture: OpenAI embeddings for semantic understanding, Weaviate for scalable similarity search, Mistral for domain classification, and custom ML for final decisions.

This hybrid approach leverages the strengths of each component while mitigating their individual limitations. The result is a system that handles production scale with production quality."

**Timing**: 60 seconds  
**Transition**: "The economic impact is equally impressive..."

---

### Slide 21: Cost-Benefit Analysis (60 seconds)
**Visual**: cost_benefit_analysis.png - ROI calculation

**Script**:
"The economics are compelling: $49,400 for automated processing versus $1.76 million for manual review - a 97% cost reduction. Beyond cost, we gain consistency, scalability, and the ability to process updates continuously.

Most importantly, this system frees human experts to focus on complex edge cases rather than routine comparisons, multiplying their impact across the entire catalog."

**Timing**: 60 seconds  
**Transition**: "This success story teaches us broader lessons..."

---

### Slide 22: Franz Schubert Success Story (60 seconds)
**Visual**: franz_schubert_decision_tree.png - Complete disambiguation

**Script**:
"Let's return to our protagonist: Franz Schubert. Our system now correctly identifies that Symphony No. 9 and Winterreise belong to the composer (same entity), while Archaeological Photography belongs to the photographer (different entity).

The combination of semantic similarity, domain classification, and temporal features provides the context that pure text similarity couldn't capture. This case study exemplifies how multiple AI techniques solve problems that individual approaches cannot."

**Timing**: 60 seconds  
**Transition**: "The broader impact extends far beyond Yale..."

---

### Slide 23: Real-World Applications (60 seconds)
**Visual**: journey_timeline.png - Generalization opportunities

**Script**:
"This approach generalizes broadly: customer deduplication in CRM systems, author disambiguation in academic databases, product catalog merging in e-commerce, medical record linking in healthcare systems.

The core insight - combining semantic similarity with domain context and temporal features - applies wherever we need to resolve entity identity across large, noisy datasets."

**Timing**: 60 seconds  
**Transition**: "What did we learn from this journey?"

---

## CONCLUSION & DISCUSSION (3 minutes, 3 slides)

### Slide 24: Key Lessons Learned (75 seconds)
**Visual**: journey_timeline.png - Decision points and learnings

**Script**:
"Our journey teaches crucial lessons for AI practitioners: Start simple and iterate based on real problems. Domain expertise drives feature engineering. Token limits matter - always test with realistic data. Vector databases enable production scale. Hot-deck imputation leverages embeddings for data quality.

Most importantly: real AI systems combine multiple techniques. No single approach - embeddings, classification, or vector search - solved our problem alone. The magic happened in the integration."

**Timing**: 75 seconds  
**Transition**: "This iterative approach is typical of production AI..."

---

### Slide 25: The Iterative Nature of AI Development (60 seconds)
**Visual**: journey_timeline.png - Evolution through challenges

**Script**:
"Notice our path: embeddings → threshold problem → domain classification → token limits → Mistral → production integration. Each challenge forced innovation. Each solution revealed new challenges.

This is how real AI systems evolve - not through grand design, but through iterative problem-solving, each step building on lessons from the previous. Your own AI projects will likely follow similar paths."

**Timing**: 60 seconds  
**Transition**: "Finally, let's consider what's next..."

---

### Slide 26: Future Directions & Your Projects (45 seconds)
**Visual**: workshop_overview.png - Applications and extensions

**Script**:
"Today's techniques are just the beginning. Future developments might include: multilingual embeddings for global catalogs, graph neural networks for relationship modeling, active learning for human feedback integration, and federated learning for privacy-preserving entity resolution.

Most importantly: how will you apply these concepts to your own research? The combination of embeddings, classification, and vector databases opens possibilities across every field represented in this room."

**Timing**: 45 seconds  
**Transition**: "Thank you for joining this journey. Questions?"

---

## TIMING SUMMARY
- **Introduction**: 4 minutes (3 slides)
- **Embeddings Fundamentals**: 8 minutes slides + 7 minutes coding = 15 minutes  
- **Domain Classification**: 8 minutes slides + 7 minutes coding = 15 minutes
- **Complete Pipeline**: 10 minutes slides + 8 minutes coding = 18 minutes
- **Conclusion**: 3 minutes (3 slides)
- **Total**: 45 minutes

## MATERIALS NEEDED
- Laptops with Google Colab access
- Three Jupyter notebooks (already created)
- All visualization files in pres/img/
- OpenAI API keys for students (optional - notebooks have mock functions)
- Mistral API keys for students (optional - notebooks have mock functions)

## PRESENTER NOTES
- Keep coding breaks focused - provide clear instructions
- Circulate during coding breaks to help students
- Franz Schubert example should thread through entire presentation
- Emphasize real-world applicability for diverse student backgrounds
- Have backup slides ready in case of technical issues
- Encourage questions throughout, not just at the end
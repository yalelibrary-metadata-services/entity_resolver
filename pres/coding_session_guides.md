# Coding Session Guides: From Words to Vectors Workshop
## Hands-On Notebook Integration

---

## CODING SESSION #1: Tokenization & Similarity Analysis
**Duration**: 7 minutes  
**Notebook**: yale_embeddings_fundamentals.ipynb  
**Timing in Presentation**: After Slide 5 (Tokenization - The Hidden Foundation)

### Learning Objectives
Students will:
- Experience how tokenization varies across languages and text types
- Calculate text embeddings using OpenAI's API
- Measure semantic similarity between different text pairs
- Discover the Franz Schubert similarity problem firsthand

### Pre-Session Setup (1 minute)
**Instructor announces**:
"Now let's get hands-on! Please open yale_embeddings_fundamentals.ipynb in Google Colab. We're going to explore cells 8 through 16. Don't worry about understanding every line of code - focus on the outputs and what they tell us."

**Students should**:
1. Open Google Colab in their browser
2. Navigate to the notebook URL (provided in handout)
3. Run the setup cells (1-4) if not already executed
4. Locate cell 8 (tokenization analysis section)

### Activity 1: Tokenization Analysis (2 minutes)
**Target Cells**: 8-10

**Instructor guidance**:
"First, let's see how tokenization works. Execute cells 8, 9, and 10 in sequence."

**What students will see**:
- Cell 8: Different text examples being tokenized
- Cell 9: Multilingual tokenization comparison showing efficiency differences
- Cell 10: Visual chart comparing token counts across languages

**Key observations to highlight**:
- Simple names: 2-4 tokens
- Full catalog records: 150-300 tokens
- Language efficiency varies dramatically (English: 3.5 chars/token, Chinese: ~2.1 chars/token)
- Realistic data often exceeds simple model limits

**Instructor circulates asking**:
- "What surprises you about these token counts?"
- "How might this affect API costs?"
- "Why do you think Chinese is less efficient?"

### Activity 2: Embedding Creation (2 minutes)
**Target Cells**: 12-14

**Instructor guidance**:
"Now let's turn text into vectors. Run cells 12, 13, and 14."

**What students will see**:
- Cell 12: Function definitions for embedding and similarity calculation
- Cell 13: Live embedding creation for Franz Schubert examples
- Cell 14: Similarity calculations between different text pairs

**Key observations to highlight**:
- Each embedding has 1,536 dimensions
- Embeddings are normalized (values roughly between -1 and 1)
- API calls create the embeddings in real-time (or use mock functions)

**Common issues and solutions**:
- API key not set: "That's fine - we have mock functions for demo purposes"
- Slow API responses: "This is normal - production systems use batch processing"
- Don't understand the math: "Focus on the similarity scores coming up"

### Activity 3: Similarity Testing (3 minutes)
**Target Cells**: 14-16

**Instructor guidance**:
"Now the exciting part - let's measure semantic similarity. Execute cells 14, 15, and 16."

**What students will see**:
- Cell 14: Similarity analysis between name variations and cross-language examples
- Cell 15: Interactive visualization of similarity scores
- Cell 16: The Franz Schubert problem demonstrated with actual similarity scores

**Critical observations for the workshop**:
- High similarity (0.95+): "Schubert, Franz" vs "Franz Schubert" 
- Cross-language similarity (0.85+): "Photography in archaeology" vs "Fotografía en arqueología"
- **The problem**: Different Franz Schuberts still score 0.7-0.8 similarity
- Threshold dilemma: where do you draw the line?

**Instructor highlights**:
"Look at the Franz Schubert scores. The composer and photographer score around 0.76. That's high enough to suggest they're the same person, but they're completely different individuals. This is the core problem that drove our entire system development."

### Session Debrief (1 minute)
**Instructor asks**:
1. "What did you notice about the Franz Schubert similarity scores?"
2. "Why might a simple threshold not work here?"
3. "What other information might help distinguish these entities?"

**Bridge to next section**:
"This similarity-but-not-identical pattern led us to realize we needed additional context beyond just text similarity. That's where domain classification comes in..."

### Troubleshooting Guide

**Common Issues**:

| Problem | Solution | Prevention |
|---------|----------|------------|
| OpenAI API key error | Point to mock function fallback | Test keys before workshop |
| Slow notebook loading | Have students share working notebooks | Prepare backup static outputs |
| Code cells won't run | Restart kernel and run from beginning | Include this in pre-session instructions |
| Students finish early | Ask them to help neighbors or test their own examples | Prepare extension activities |
| Students confused by code | Emphasize focusing on outputs not code details | Set expectations clearly |

**Extension Activities** (for fast finishers):
- Modify the text examples to test your own research domain
- Try calculating similarity between concepts from your field
- Experiment with different languages in the multilingual analysis

---

## CODING SESSION #2: SetFit vs Mistral Comparison
**Duration**: 7 minutes  
**Notebook**: yale_domain_classification.ipynb  
**Timing in Presentation**: After Slide 12 (Mistral Classifier Factory - The Solution)

### Learning Objectives
Students will:
- Experience SetFit's token length limitations with realistic data
- Test Mistral's ability to handle full catalog records
- Witness domain classification successfully distinguishing Franz Schuberts
- Understand why architectural choices matter in production systems

### Pre-Session Setup (1 minute)
**Instructor announces**:
"Time for our second coding session! Open yale_domain_classification.ipynb. We'll compare two approaches to domain classification and see why we had to pivot from our original plan."

**Students should**:
1. Switch to the domain classification notebook
2. Ensure cells 1-7 have been run (setup and data loading)
3. Navigate to cell 14 (token length analysis)

### Activity 1: Token Length Problem Discovery (2 minutes)
**Target Cells**: 14-16

**Instructor guidance**:
"First, let's discover the problem that broke our SetFit approach. Run cells 14, 15, and 16."

**What students will see**:
- Cell 14: Token length analysis showing realistic catalog records
- Cell 15: Visualization of token counts vs. model limits
- Cell 16: SetFit limitations demonstrated with actual data

**Key observations to highlight**:
- Simple examples: 15-45 tokens (within SetFit's 128-token limit)
- Realistic records: 150-300 tokens (exceed SetFit's limit)
- Only 25% of real data would fit in SetFit
- Truncation loses crucial context for classification

**Instructor emphasizes**:
"This is a perfect example of why you must test with realistic data, not toy examples. SetFit looked perfect in theory but failed with our actual catalog records."

**Discussion questions**:
- "How would you feel about losing 75% of your data to token limits?"
- "What would happen if we truncated records to fit?"
- "Why might longer context be important for domain classification?"

### Activity 2: Mistral Alternative Testing (3 minutes)
**Target Cells**: 18-20

**Instructor guidance**:
"Now let's see how Mistral handles the same challenge. Execute cells 18, 19, and 20."

**What students will see**:
- Cell 18: Mistral classification setup and prompt engineering
- Cell 19: Testing Mistral with full-length catalog records
- Cell 20: Comparison of SetFit vs Mistral capabilities

**Key observations to highlight**:
- Mistral handles 32,000 tokens (vs SetFit's 128)
- Processes complete catalog records without truncation
- Maintains context for accurate domain classification
- No training required - works with few-shot examples

**Critical demonstration**:
"Watch how Mistral classifies our Franz Schubert records. The composer's symphony gets classified as 'Music, Sound, and Sonic Arts' while the photographer's archaeology book gets 'Documentary and Technical Arts'. This domain difference becomes a powerful feature for entity resolution."

### Activity 3: Domain Classification Success (2 minutes)
**Target Cells**: 20-22

**Instructor guidance**:
"Let's see domain classification solve our Franz Schubert problem. Run cells 20, 21, and 22."

**What students will see**:
- Cell 20: Franz Schubert records being classified into different domains
- Cell 21: Comparison matrix showing classification results
- Cell 22: Feature engineering showing how domain dissimilarity works

**Breakthrough moment**:
Students should see the "aha!" moment where domain classification provides the missing context that pure text similarity couldn't capture.

**Instructor highlights**:
"This is our breakthrough! When two records have the same person name but different domains, we now know they're likely different people. Domain dissimilarity becomes a powerful feature that solves the threshold problem."

### Session Debrief (1 minute)
**Instructor asks**:
1. "What broke when we tried SetFit with realistic data?"
2. "How did Mistral solve this problem?"
3. "How does domain classification help distinguish the Franz Schuberts?"

**Bridge to next section**:
"Now we have two powerful tools: semantic similarity from embeddings and contextual understanding from domain classification. Our final session shows how we combined these with other features in a complete production system."

### Troubleshooting Guide

**Common Issues**:

| Problem | Solution | Prevention |
|---------|----------|------------|
| SetFit model download fails | Use pre-computed examples in text | Test downloads before workshop |
| Mistral API not available | Use mock classification functions | Prepare fallback responses |
| Students don't see the problem | Emphasize the 75% data loss statistic | Make the impact concrete |
| Classification seems arbitrary | Explain the taxonomy development process | Connect to real-world categorization |
| Code runs slowly | Use cached results or mock functions | Optimize for workshop timing |

**Extension Activities**:
- Classify examples from your own research domain
- Design taxonomy categories for your field
- Test edge cases where domain classification might struggle

---

## CODING SESSION #3: Complete Pipeline Demo
**Duration**: 8 minutes  
**Notebook**: yale_entity_resolution_pipeline.ipynb  
**Timing in Presentation**: After Slide 18 (Feature Engineering for Entity Resolution)

### Learning Objectives
Students will:
- Experience vector similarity search for finding related records
- Witness vector hot-deck imputation intelligently filling missing data
- See multi-feature classification combining all techniques
- Observe successful Franz Schubert disambiguation in the complete system
- Understand how production AI systems integrate multiple components

### Pre-Session Setup (1 minute)
**Instructor announces**:
"For our final coding session, you'll experience the complete entity resolution pipeline that processes millions of records at Yale. Open yale_entity_resolution_pipeline.ipynb - this brings everything together."

**Students should**:
1. Switch to the pipeline notebook
2. Ensure setup cells (1-8) have been executed
3. Navigate to cell 10 (vector similarity search)

### Activity 1: Vector Similarity Search (2 minutes)
**Target Cells**: 10-12

**Instructor guidance**:
"First, let's see how vector databases find similar records efficiently. Run cells 10, 11, and 12."

**What students will see**:
- Cell 10: Vector similarity search finding related records
- Cell 11: Similarity scores and ranking of candidate matches
- Cell 12: Analysis of which records are most similar to a query

**Key observations to highlight**:
- Instant similarity search across thousands of records
- Semantic similarity revealed through vector search
- Related entities cluster together in vector space
- This replaces exhaustive pairwise comparisons

**Instructor emphasizes**:
"This is how we avoid 155 trillion comparisons. Instead of comparing every record to every other record, we use vector similarity to find the most likely candidates instantly."

**Discussion points**:
- "Notice how the most similar records are often from the same entity"
- "This is like semantic search - finding meaning, not just keywords"
- "Vector databases make this scale to millions of records"

### Activity 2: Vector Hot-Deck Imputation (3 minutes)
**Target Cells**: 12-14

**Instructor guidance**:
"Now let's see something revolutionary - using vector similarity to fill missing data. Execute cells 12, 13, and 14."

**What students will see**:
- Cell 12: Records with missing subject fields identified
- Cell 13: Vector hot-deck imputation finding semantic donors
- Cell 14: Missing fields filled with contextually appropriate values

**Key observations to highlight**:
- Missing subjects filled by finding semantically similar records
- "Hot-deck" donors chosen based on semantic similarity, not random chance
- Imputed values are contextually appropriate for the record type
- Data quality improvement through intelligent imputation

**Innovation moment**:
"This is a novel application of embeddings - instead of just measuring similarity, we're using semantic understanding to enhance data quality. Traditional hot-deck imputation uses statistical similarity; vector hot-deck uses semantic similarity."

**Instructor highlights**:
- Show examples of successfully imputed subject fields
- Point out how donor records are semantically related
- Emphasize the intelligence of context-aware imputation

### Activity 3: Multi-Feature Classification (2 minutes)
**Target Cells**: 14-16

**Instructor guidance**:
"Now we combine everything - embeddings, domain classification, and additional features. Run cells 14, 15, and 16."

**What students will see**:
- Cell 14: Feature engineering combining all signals
- Cell 15: Logistic regression classifier training
- Cell 16: Feature importance analysis showing which features matter most

**Key observations to highlight**:
- Five features capture different aspects of entity similarity
- Domain dissimilarity has strong negative weight (different domains = different people)
- Person similarity has positive weight (similar names = same person)
- Classifier learns optimal combination automatically

**Technical insight**:
"Notice how the classifier balances multiple signals. High person similarity + same domain + matching dates = strong match. High person similarity + different domains = likely different people."

### Activity 4: Franz Schubert Final Test (1 minute)
**Target Cells**: 19-20

**Instructor guidance**:
"The moment of truth - let's see if our complete system correctly distinguishes the two Franz Schuberts. Run cells 19 and 20."

**What students will see**:
- Cell 19: Franz Schubert pairs being classified by the complete system
- Cell 20: Detailed breakdown of feature contributions for each decision

**Climactic moment**:
Students should see the system correctly identifying:
- Composer records as matching each other (same entity)
- Photographer records as matching each other (same entity)  
- Composer vs photographer as different entities

**Victory celebration**:
"Success! The system that started with a simple similarity problem now correctly distinguishes between Franz Schubert the composer and Franz Schubert the photographer. It combines semantic similarity, domain classification, temporal features, and more."

### Session Debrief (1 minute)
**Instructor asks**:
1. "How did the complete system solve the Franz Schubert problem?"
2. "Which features were most important for the classification decisions?"
3. "What impressed you most about the vector hot-deck imputation?"

**Final bridge**:
"You've now experienced a complete production AI system that processes 17.6 million records. This combination of techniques achieves 99.55% precision in production at Yale."

### Troubleshooting Guide

**Common Issues**:

| Problem | Solution | Prevention |
|---------|----------|------------|
| Vector search too slow | Use pre-computed similarity results | Optimize data size for workshop |
| Imputation doesn't work | Show static examples of successful cases | Prepare backup demonstrations |
| Feature engineering confusing | Focus on intuitive explanations | Emphasize conceptual understanding |
| Classification results unclear | Highlight the key Franz Schubert distinctions | Prepare clear examples |
| Students overwhelmed by complexity | Emphasize that they understand the components | Break down into familiar pieces |

**Extension Activities**:
- Test the classifier on new entity pairs
- Modify feature weights to see how decisions change
- Explore the imputation results for different record types

---

## GENERAL CODING SESSION MANAGEMENT

### Pre-Workshop Preparation
1. **Test all notebooks** in fresh Google Colab environments
2. **Verify API keys** work or prepare mock function fallbacks
3. **Time each coding session** to ensure realistic estimates
4. **Prepare backup static outputs** for each key cell
5. **Create shared folder** with notebook links and resources

### During Sessions - Instructor Checklist
- [ ] Circulate constantly - don't stay at the front
- [ ] Help struggling students but don't solve everything for them
- [ ] Note common issues to address in debrief
- [ ] Keep energy high and encourage collaboration
- [ ] Use timer to manage session length
- [ ] Take photos/screenshots of interesting student discoveries

### Student Support Strategies
- **Pair programming**: Encourage students to work together
- **Fast finisher helpers**: Students who complete early help others
- **Backup plans**: Static outputs if live code fails
- **Focus guidance**: "Don't worry about understanding every line - focus on the outputs"

### Learning Assessment
After each session, students should be able to:

**Session 1**:
- Explain why tokenization matters for AI systems
- Describe the threshold problem in entity resolution
- Recognize that semantic similarity alone isn't sufficient

**Session 2**:
- Compare model capabilities and limitations
- Understand how domain classification provides context
- Appreciate the importance of testing with realistic data

**Session 3**:
- Describe how multiple AI techniques combine in production systems
- Explain vector hot-deck imputation innovation
- Understand end-to-end entity resolution workflows

### Success Metrics
- **Engagement**: Students actively running code and asking questions
- **Understanding**: Students can explain the Franz Schubert problem and solution
- **Confidence**: Students feel they could apply these techniques to their own problems
- **Integration**: Students see how the sessions build on each other

### Common Questions and Answers

**Q: "Do we need to understand all the code?"**
A: "No - focus on the inputs, outputs, and what they tell us. The concepts matter more than implementation details."

**Q: "What if the code doesn't work?"**
A: "That's realistic AI development! We have backup examples to show what should happen."

**Q: "Can we use this for our own research?"**
A: "Absolutely! These notebooks are yours to modify and adapt."

**Q: "How do you know if your approach is working?"**
A: "Great question - that's why we measure precision, recall, and test with real data."

### Post-Session Follow-Up
1. **Share all notebooks** with students for further exploration
2. **Provide additional resources** for continued learning
3. **Offer office hours** for deeper questions
4. **Document common issues** for future workshops
5. **Collect feedback** on difficulty level and pacing
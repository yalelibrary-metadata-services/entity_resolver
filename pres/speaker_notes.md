# Speaker Notes: From Words to Vectors Workshop
## Detailed Presenter Guidance

### PRE-WORKSHOP SETUP (15 minutes before start)

#### Technical Setup
- [ ] Test projector with all visualization files
- [ ] Verify Google Colab access on demo laptop
- [ ] Load all three notebooks: yale_embeddings_fundamentals.ipynb, yale_domain_classification.ipynb, yale_entity_resolution_pipeline.ipynb
- [ ] Test OpenAI API connection (use backup mock functions if needed)
- [ ] Have backup slides ready for technical issues

#### Room Setup
- [ ] Arrange seating for laptop use during coding breaks
- [ ] Test microphone and audio
- [ ] Prepare handout with notebook URLs and API key instructions
- [ ] Set up timer/clock visible to presenter

#### Mental Preparation
- [ ] Review Franz Schubert examples - this is your central thread
- [ ] Practice transitions between conceptual slides and coding breaks
- [ ] Prepare 2-3 backup questions for slow moments
- [ ] Remember: this is a journey story, not just a technical presentation

---

## INTRODUCTION & SETUP (4 minutes)

### Slide 1: Welcome & Overview
**Energy Level**: High enthusiasm - set the tone  
**Key Message**: This is a journey story with real challenges and breakthroughs

**Presenter Notes**:
- Smile and make eye contact - graduate students need to feel welcome
- Emphasize "journey" - this isn't just theory, it's how we actually built something
- Point out diverse backgrounds: "Whether you're in literature, physics, or computer science, entity resolution affects your field"
- Preview the hands-on nature: "You'll write code, not just listen"

**Audience Engagement**:
- Ask: "How many of you have worked with library catalogs or bibliographic data?"
- Note responses - tailor examples to their experience level

**Timing Cues**:
- 30 seconds: Introduction and welcome
- 60 seconds: Overview of learning objectives  
- 90 seconds: Transition to Franz Schubert

### Slide 2: The Franz Schubert Problem
**Energy Level**: Build intrigue and mystery  
**Key Message**: Simple problems often have complex solutions

**Presenter Notes**:
- This is your hook - make Franz Schubert memorable
- Use physical gestures: point to different parts of screen for each Franz
- Emphasize the dates: "1797-1828 versus 1930-1989 - completely different centuries!"
- Make it relatable: "Imagine trying to separate your records from someone else with your exact name"

**Technical Notes**:
- If visualization doesn't load, have the text examples ready to read aloud
- Be prepared to explain why this matters for digital humanities

**Common Questions**:
- Q: "Why not just use dates to distinguish?"
- A: "Great question - we'll see that birth/death data is often missing or inconsistent"

### Slide 3: The Journey Ahead
**Energy Level**: Build anticipation  
**Key Message**: Clear roadmap with hands-on learning

**Presenter Notes**:
- This is your roadmap slide - students should understand the structure
- Emphasize the coding breaks: "Every concept we discuss, you'll implement"
- Preview the progression: "We start simple and build complexity based on real challenges"
- Set expectations: "Don't worry about perfect understanding now - it will build"

**Timing Management**:
- Keep this slide brief - students are eager to start
- Transition smoothly: "Let's dive into our first challenge..."

---

## PART 1: EMBEDDINGS FUNDAMENTALS (15 minutes)

### Slide 4: The Evolution of Text Embeddings
**Energy Level**: Moderate - educational building  
**Key Message**: Stand on the shoulders of giants

**Presenter Notes**:
- This provides necessary context but shouldn't overwhelm
- Focus on the progression: "Each generation solved problems of the previous"
- Highlight our choice: "text-embedding-3-small because it's designed for similarity tasks"
- Connect to their experience: "You've probably used these models without knowing it"

**Adaptation for Audience**:
- For CS students: Mention attention mechanisms and transformer architecture
- For humanities students: Focus on practical capabilities and applications
- For science students: Emphasize the empirical improvement over time

### Slide 5: Tokenization - The Hidden Foundation
**Energy Level**: Increase interest - this surprises people  
**Key Message**: The invisible layer that affects everything

**Presenter Notes**:
- Most people don't think about tokenization - make it visible
- Use multilingual examples: "Look how English is more efficient than Chinese"
- Connect to costs: "This knowledge saved us thousands of dollars"
- Preview coding: "In our first coding break, you'll see this in action"

**Interactive Moment**:
- Ask: "Any guesses why Chinese might be less token-efficient?"
- Accept answers, then explain character-based vs. word-based tokenization

### CODING BREAK #1: Tokenization & Similarity (7 minutes)
**Energy Level**: High engagement - first hands-on moment  
**Your Role**: Facilitator and troubleshooter

**Pre-Break Instructions** (1 minute):
1. "Open yale_embeddings_fundamentals.ipynb in Google Colab"
2. "We're focusing on cells 8-16"
3. "Don't worry if you don't understand every line - focus on the outputs"
4. "Raise your hand if you hit any errors"

**During Break - Circulate and Help**:
- Common issue: API key not set - point to mock functions
- Watch for students who finish early - have them help others
- Note interesting observations from students - use in debrief

**Debrief Questions** (1 minute):
- "What surprised you about the tokenization results?"
- "What did you notice about the Franz Schubert similarity scores?"
- Bridge to next section: "This moderate similarity is our core challenge"

### Slide 6: Semantic Similarity in Action
**Energy Level**: Build on coding excitement  
**Key Message**: Embeddings work well but have limitations

**Presenter Notes**:
- Reference what students just saw: "You just calculated these similarities"
- Highlight the success: "0.98 for name variations - nearly perfect"
- Emphasize the problem: "But 0.76 for different Franz Schuberts - too high"
- Use their data: "What scores did you see in your notebooks?"

### Slide 7: The Threshold Problem
**Energy Level**: Building tension - the plot thickens  
**Key Message**: No silver bullet solutions in real AI

**Presenter Notes**:
- This is a key insight - emphasize the universality
- Use concrete numbers: "We tested every threshold from 0.5 to 0.95"
- Make it relatable: "It's like setting a universal height for doorways - won't work"
- Foreshadow solution: "This insight drove everything we built next"

**Pedagogical Note**:
- Students often expect AI to have simple solutions
- This slide teaches the complexity of real-world AI problems

### Slide 8: Cost Analysis - Real-World Constraints
**Energy Level**: Practical reality check  
**Key Message**: Economics drive architecture in production AI

**Presenter Notes**:
- Ground them in reality: "Cool technology means nothing if you can't afford it"
- Use specific numbers: "$52,800 versus $26,400 - batch processing matters"
- Connect to their futures: "You'll face these constraints in any real project"
- Transition: "With these constraints, we needed smarter features"

**For Different Audiences**:
- Business students: Emphasize ROI calculations
- CS students: Note how constraints drive architectural decisions
- Humanities students: Connect to grant funding realities

---

## PART 2: DOMAIN CLASSIFICATION (15 minutes)

### Slide 9: Why Domain Context Matters
**Energy Level**: Breakthrough moment - the "aha" insight  
**Key Message**: Context solves what similarity alone cannot

**Presenter Notes**:
- This is your breakthrough slide - build excitement
- Use dramatic contrast: "Music versus Photography - completely different domains"
- Connect to human cognition: "This is how humans distinguish them"
- Show the taxonomy: "We mapped all human knowledge domains"

**Engage the Audience**:
- Ask: "How would you classify your own research domain?"
- Note their responses - use as examples later

### Slide 10: SetFit - The Initial Choice
**Energy Level**: Build anticipation for the coming problem  
**Key Message**: Good ideas can have fatal limitations

**Presenter Notes**:
- Don't spend too long here - this is setup for the problem
- Emphasize why it seemed perfect: "Few-shot learning, fast training"
- Build tension: "We were excited... until we hit a wall"
- Quick transition: "Let's see what broke our approach"

### Slide 11: The Token Length Discovery
**Energy Level**: Problem revealed - dramatic tension  
**Key Message**: Real data breaks theoretical approaches

**Presenter Notes**:
- This is a teachable moment about real vs. theoretical constraints
- Use specific numbers: "128 tokens versus 200-300 for real records"
- Show the math: "Only 25% of our data would fit"
- Emphasize the lesson: "Always test with realistic data"

**Teaching Moment**:
- Students learn that academic papers don't always match real-world constraints
- Emphasize the iterative nature of AI development

### Slide 12: Mistral Classifier Factory - The Solution
**Energy Level**: Relief and excitement - problem solved  
**Key Message**: Right tool for the right job

**Presenter Notes**:
- Build the contrast: "32,000 tokens versus 128 - game changer"
- Emphasize practical benefits: "No training, handles our complete records"
- Show the economics: "$0.001 per classification - affordable"
- Preview coding: "Let's test both approaches"

### CODING BREAK #2: SetFit vs Mistral (7 minutes)
**Energy Level**: High engagement - they're testing the solution  
**Your Role**: Guide discovery of the key insight

**Pre-Break Instructions** (1 minute):
1. "Open yale_domain_classification.ipynb"
2. "Focus on cells 14-20 - comparing both approaches"
3. "Pay attention to which approach handles full records"
4. "Notice how domain classification distinguishes Franz Schuberts"

**During Break**:
- Help with token length analysis - students should see the limitation
- Point out successful domain classifications
- Note when students discover the key insight

**Debrief Questions** (1 minute):
- "What happened when you tried long records with SetFit?"
- "How did Mistral handle the same records?"
- "What domains did the different Franz Schuberts get classified as?"

### Slide 13: Taxonomy in Action
**Energy Level**: Success celebration  
**Key Message**: Domain classification solves the Franz Schubert problem

**Presenter Notes**:
- Connect to their coding experience: "You just saw this work"
- Highlight the breakthrough: "Different domains = different people"
- Show the feature: "Domain dissimilarity becomes a powerful classification feature"
- Build toward integration: "But we needed to combine this with other features"

### Slide 14: Performance Analysis
**Energy Level**: Analytical - show the data  
**Key Message**: Data-driven decisions in AI development

**Presenter Notes**:
- Keep this brief - the data speaks for itself
- Highlight key metrics: "100% versus 75% data coverage"
- Show the economics: "$18K versus $1.76M - clear ROI"
- Quick transition: "With classification solved, we needed integration"

### Slide 15: Integration Strategy
**Energy Level**: Building toward the climax  
**Key Message**: Real systems combine multiple techniques

**Presenter Notes**:
- Summarize progress: "Embeddings + domain classification"
- Preview complexity: "Production systems need more"
- List what's coming: "Vector databases, data imputation, feature engineering"
- Build anticipation: "Our final phase combines everything"

---

## PART 3: COMPLETE PIPELINE (18 minutes)

### Slide 16: Vector Databases - Scaling Similarity Search
**Energy Level**: Technical excitement - the engineering solution  
**Key Message**: Scale requires smart algorithms

**Presenter Notes**:
- Start with the impossible: "155 trillion comparisons - impossible"
- Show the solution: "99.23% reduction through smart indexing"
- Explain Weaviate: "Production-scale similarity search"
- Connect to their experience: "Like Google search but for semantic similarity"

**Technical Adaptation**:
- For CS students: Mention HNSW indexing and approximate nearest neighbors
- For others: Focus on the scale and speed benefits

### Slide 17: Vector Hot-Deck Imputation
**Energy Level**: Innovation excitement - this is novel  
**Key Message**: Semantic similarity enables intelligent data enhancement

**Presenter Notes**:
- Explain traditional hot-deck: "Statistical similarity for missing values"
- Show the innovation: "Vector hot-deck uses semantic similarity"
- Make it concrete: "Find records about similar topics, copy their subjects"
- Emphasize intelligence: "Not random - contextually aware"

**Novel Concept Alert**:
- This may be unfamiliar - use analogies
- "Like asking someone with similar interests for recommendations"

### Slide 18: Feature Engineering for Entity Resolution
**Energy Level**: Technical mastery - bringing it all together  
**Key Message**: Multiple features capture different aspects of similarity

**Presenter Notes**:
- List all features: "Name, content, interaction, domain, temporal"
- Show the weights: "Domain dissimilarity has high negative weight"
- Connect to Franz Schubert: "This solves our original problem"
- Preview integration: "Let's see it all work together"

### CODING BREAK #3: Complete Pipeline Demo (8 minutes)
**Energy Level**: Climax - everything comes together  
**Your Role**: Guide them through the complete solution

**Pre-Break Instructions** (1 minute):
1. "Open yale_entity_resolution_pipeline.ipynb"
2. "Focus on cells 12-24 - the complete pipeline"
3. "Watch for Franz Schubert disambiguation"
4. "Notice how all features combine for final decisions"

**During Break - Key Moments to Highlight**:
- Vector similarity finding related records
- Hot-deck imputation filling missing data
- Feature engineering combining all signals
- Franz Schubert successful disambiguation

**Debrief Questions** (1 minute):
- "Did the system correctly distinguish the two Franz Schuberts?"
- "Which features were most important for that decision?"
- "What impressed you most about the complete pipeline?"

### Slide 19: Production Performance Metrics
**Energy Level**: Victory lap - show the success  
**Key Message**: Real systems achieve real results

**Presenter Notes**:
- Lead with the headline: "99.55% precision - extraordinary accuracy"
- Show the scale: "17.6 million records, 14,930 test pairs"
- Emphasize impact: "99.23% reduction in manual work"
- Connect to their coding: "You just saw how this works"

**Impact Emphasis**:
- This isn't academic - it's production scale
- Real librarians use this system daily

### Slide 20: Computational Efficiency
**Energy Level**: Technical appreciation  
**Key Message**: Modern AI architecture enables impossible tasks

**Presenter Notes**:
- Show the architecture: "OpenAI + Weaviate + Mistral + custom ML"
- Emphasize hybrid approach: "Leverage strengths, mitigate weaknesses"
- Connect to broader AI: "This is how production systems are built"
- Preview economics: "The cost analysis is equally impressive"

### Slide 21: Cost-Benefit Analysis
**Energy Level**: Business case excitement  
**Key Message**: AI delivers transformational ROI

**Presenter Notes**:
- Lead with savings: "97% cost reduction - $1.76M to $49K"
- Show broader benefits: "Consistency, scalability, continuous processing"
- Emphasize human augmentation: "Experts focus on complex cases"
- Connect to their futures: "These economics drive AI adoption"

### Slide 22: Franz Schubert Success Story
**Energy Level**: Emotional satisfaction - full circle  
**Key Message**: We solved the problem that started our journey

**Presenter Notes**:
- Return to the beginning: "Remember our protagonist Franz Schubert?"
- Show the solution: "Composer and photographer correctly distinguished"
- List the techniques: "Semantic + domain + temporal features"
- Celebrate the insight: "Multiple AI techniques solve what single approaches cannot"

**Narrative Closure**:
- This completes the story arc
- Students should feel the satisfaction of problem solved

### Slide 23: Real-World Applications
**Energy Level**: Expanding horizons  
**Key Message**: These techniques generalize broadly

**Presenter Notes**:
- List applications: "CRM, academic databases, e-commerce, healthcare"
- Emphasize the pattern: "Semantic similarity + domain context + temporal features"
- Connect to their fields: "How might this apply to your research?"
- Build confidence: "You now understand production AI systems"

**Interactive Moment**:
- Ask: "What entity resolution challenges exist in your field?"
- Collect 2-3 examples - validate their applicability

---

## CONCLUSION & DISCUSSION (3 minutes)

### Slide 24: Key Lessons Learned
**Energy Level**: Reflective wisdom  
**Key Message**: Practical AI development principles

**Presenter Notes**:
- Emphasize iteration: "Start simple, iterate based on real problems"
- Highlight domain expertise: "Technical skills + domain knowledge"
- Stress testing: "Always use realistic data"
- Show integration: "Real systems combine multiple techniques"

**Teaching Moment**:
- These lessons apply beyond entity resolution
- Universal principles for AI development

### Slide 25: The Iterative Nature of AI Development
**Energy Level**: Philosophical reflection  
**Key Message**: Real AI is messy and iterative

**Presenter Notes**:
- Show the path: "Embeddings → threshold problem → domain classification → production"
- Normalize the messiness: "This is how real AI development works"
- Encourage resilience: "Challenges force innovation"
- Prepare them: "Your projects will follow similar paths"

**Confidence Building**:
- Students should feel prepared for real AI challenges
- Emphasize that iteration is normal, not failure

### Slide 26: Future Directions & Your Projects
**Energy Level**: Inspirational send-off  
**Key Message**: You're ready to build the next generation

**Presenter Notes**:
- List future possibilities: "Multilingual, graph networks, active learning"
- Connect to their work: "How will you apply these concepts?"
- Show opportunity: "Every field needs entity resolution"
- End with confidence: "You have the tools and understanding"

**Final Question**:
- "What's the first entity resolution challenge you want to tackle?"
- Take 2-3 answers, validate their approaches

---

## Q&A MANAGEMENT

### Likely Questions and Responses

**Q: "How do you handle name variations across cultures?"**
A: "Great question - our multilingual tokenization analysis showed this challenge. We rely on OpenAI's embeddings, which handle many name variations well, plus domain classification for disambiguation. Culture-specific preprocessing could improve this further."

**Q: "What about privacy concerns with API-based embeddings?"**
A: "Critical consideration. For sensitive data, you'd use local embeddings like sentence-transformers. The trade-off is performance vs. privacy. We chose OpenAI for public catalog data."

**Q: "Could this work for smaller datasets?"**
A: "Absolutely. The techniques scale down well. Even with 1,000 records, domain classification and feature engineering provide value. Vector databases are overkill for small datasets - simple similarity matrices work."

**Q: "How do you keep the taxonomy up to date?"**
A: "Domain evolution is ongoing. We use a combination of expert review and automated analysis of new classifications to identify taxonomy gaps. It's similar to maintaining any controlled vocabulary."

**Q: "What happens when the classifier makes mistakes?"**
A: "We have a human review workflow for low-confidence predictions and user feedback systems. The key is designing for graceful failure - when uncertain, flag for human review rather than making wrong decisions."

### If Technology Fails

**Backup Plan 1**: Use static screenshots of notebook outputs
**Backup Plan 2**: Talk through the code logic without execution
**Backup Plan 3**: Focus on conceptual understanding and defer hands-on to later

### Time Management

**Running Over Time**:
- Skip Slide 14 (performance analysis details)
- Shorten coding breaks by 1-2 minutes each
- Combine slides 20-21 (efficiency and cost analysis)

**Running Under Time**:
- Extend Q&A discussion
- Add more detail to technical explanations
- Take additional questions during coding breaks

### Energy Management

**If Energy Drops**:
- Add more interaction: "Turn to your neighbor and discuss..."
- Use physical movement: "Stand up if you've worked with..."
- Increase pace and enthusiasm in voice

**Maintain Engagement**:
- Reference their coding experiences throughout
- Use their names when they ask questions
- Connect examples to their research areas

### Final Success Metrics

Students should leave feeling:
- **Confident**: They understand production AI development
- **Inspired**: They see applications to their own work  
- **Equipped**: They have practical tools and techniques
- **Connected**: They understand how their field fits into AI

---

## POST-WORKSHOP FOLLOW-UP

### Immediately After (5 minutes)
- Collect feedback on difficulty level and pacing
- Note questions that came up repeatedly - address in follow-up materials
- Exchange contact information with interested students

### Within 24 Hours
- Send follow-up email with:
  - Links to all notebooks and materials
  - Additional resources for continued learning
  - Invitation to reach out with questions

### Within 1 Week  
- Review feedback and iterate on materials
- Consider follow-up workshop on advanced topics
- Document lessons learned for future presentations
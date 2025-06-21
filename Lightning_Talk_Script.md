# Lightning Talk Script: Entity Resolution Pipeline
## 5-Minute Delivery Script

---

## **SLIDE 1: Title Slide** (15 seconds)
### **What to Say:**
*"Good [morning/afternoon] everyone. I'm here to talk about scaling entity resolution for academic libraries - specifically how we built a machine learning pipeline for Yale's library catalog that achieves 99.55% precision while processing over 17 million name occurrences."*

### **Key Points:**
- Quick, confident introduction
- Immediately establish scale and performance
- Set expectation for technical content

---

## **SLIDE 2: The Problem & Scale** (60 seconds)
### **What to Say:**
*"Let me start with the scale of the challenge. Yale's library catalog contains 17.6 million name occurrences across 4.8 million distinct name variations. Here's a real example..."*

**[Point to examples on slide]**

*"All of these refer to the same composer - Richard Strauss. But traditional catalog systems treat them as separate entities. This fragments the scholarly record and makes research discovery nearly impossible. Manual curation at this scale simply cannot work - we needed an automated solution that could handle millions of records while maintaining library-quality precision."*

### **Key Points:**
- Use concrete examples (Richard Strauss variants)
- Emphasize impossibility of manual approach
- Connect to researcher impact

---

## **SLIDE 3: Technical Architecture** (75 seconds)
### **What to Say:**
*"Our solution is a modern machine learning pipeline with five core stages."*

**[Walk through diagram left to right]**

*"We start with preprocessing that uses hash-based deduplication to reduce computational load. Then we generate vector embeddings using OpenAI's text-embedding-3-small model and index them in Weaviate for similarity search."*

*"The innovation happens in feature engineering - we designed five critical features that capture different aspects of entity similarity. These feed into a gradient boosting classifier that we trained on carefully labeled data."*

*"Finally, we generate clusters and comprehensive reports. The entire system is containerized with Docker and designed for production deployment. What makes this special is the standardized scaling approach that ensures consistent behavior between training and production environments."*

### **Key Points:**
- Visual diagram helps audience follow
- Mention specific technologies for credibility
- Emphasize production-ready nature

---

## **SLIDE 4: Feature Engineering Innovation** (60 seconds)
### **What to Say:**
*"Let me highlight our five engineered features, because this is where domain expertise really matters."*

**[Go through each feature]**

*"Person cosine similarity uses vector embeddings to capture semantic relationships that go beyond simple string matching. The person-title squared feature combines name and work similarity using harmonic means, squared to emphasize high-confidence matches."*

*"Composite cosine similarity looks at the full record context including roles and subjects. Taxonomy dissimilarity prevents false positives across different domains - we don't want to merge a musician with a politician who happens to have a similar name."*

*"And birth-death matching uses regex extraction to validate biographical dates. The key innovation is our feature group-specific normalization that preserves binary features while optimizing continuous ones."*

### **Key Points:**
- Technical depth shows sophistication
- Explain why each feature matters
- Connect to real-world entity resolution challenges

---

## **SLIDE 5: Outstanding Results** (75 seconds)
### **What to Say:**
*"The results speak for themselves."*

**[Point to metrics table]**

*"99.55% precision means only 45 false positives out of 10,000 positive predictions. This is exceptional for entity resolution. 82% recall captures most true matches, giving us a 90% F1 score - an excellent balance."*

**[Point to confusion matrix]**

*"Our confusion matrix shows the quality - out of nearly 15,000 test pairs, we have only 45 false positives. The 99.18% AUC demonstrates near-perfect discrimination ability."*

*"We tested on 14,930 carefully labeled pairs with a decision threshold optimized for precision. These aren't toy examples - this is real-world performance on Yale's actual catalog data."*

### **Key Points:**
- Numbers are impressive - let them shine
- Emphasize low false positive rate
- Connect to real-world testing

---

## **SLIDE 6: Real-World Impact & Efficiency** (60 seconds)
### **What to Say:**
*"This translates to real operational impact. We resolved 2,535 entities into 262 final clusters with an average of 9.7 entities per cluster. The largest cluster correctly unified 92 variants of Richard Strauss."*

*"But the efficiency gains are remarkable - we achieved a 99.23% reduction in required comparisons. Instead of 10.9 billion theoretical pairwise comparisons, we need only 316 million through smart ANN clustering plus machine learning classification."*

*"The quality improvements are tangible. Franz Schubert's 89 catalog entries are now properly unified. Our cross-validation shows robust performance across diverse entity types, and the deterministic processing ensures reproducible results for production use."*

*"This enables automated catalog enhancement, improved researcher experience, and scales to millions of records with continuous learning capability."*

### **Key Points:**
- Concrete examples (Franz Schubert, Richard Strauss)
- Emphasize efficiency gains
- Connect to operational benefits

---

## **SLIDE 7: Innovation & Future Work** (45 seconds)
### **What to Say:**
*"Our key technical innovations include standardized feature scaling that maintains consistency between training and production, novel binary feature preservation, and a modular architecture where each component is independently testable."*

*"We've built in performance optimizations like multi-threaded processing, checkpoint-based resumption for fault tolerance, and smart caching for API efficiency."*

*"Looking ahead, we see cross-institutional deployment across library consortiums, real-time processing for live catalog updates, and advanced features using co-author networks and citation patterns. We're exploring active learning for continuous improvement and federated learning across library systems."*

*"This isn't just solving entity resolution - we're reimagining how academic libraries can leverage AI to serve researchers better."*

### **Key Points:**
- Quick technical highlights
- Paint vision of broader impact
- End on forward-looking note

---

## **CONCLUSION** (30 seconds)
### **What to Say:**
*"To summarize: we've achieved 99.55% precision entity resolution at academic library scale, processing 17.6 million records with 99% computational efficiency. This is a production-ready pipeline with an open architecture designed for cross-institutional deployment."*

*"We're transforming academic discovery by enabling researchers to find connections that were previously invisible. From 4.8 million name variants to unified scholarly identity - that's the power of applied AI in digital humanities."*

*"I'm happy to take any questions."*

### **Key Points:**
- Recap key achievements
- End with impact statement
- Open for questions confidently

---

## **DELIVERY TIPS**

### **Pacing:**
- **Slide 1:** Quick and confident
- **Slides 2-3:** Measured pace, let people absorb scale
- **Slide 4:** Technical but accessible
- **Slide 5:** Let numbers speak - pause after key metrics
- **Slides 6-7:** Building excitement toward conclusion

### **Body Language:**
- Stand confidently, use gestures to point to slide elements
- Make eye contact during key metrics
- Move naturally but don't pace
- Use hands to illustrate pipeline flow

### **Voice:**
- Start strong and clear
- Emphasize numbers with slight pause: "ninety-nine point five five percent"
- Build energy toward conclusion
- End with confident availability for questions

### **Backup for Technical Questions:**
- Have feature importance chart ready
- Know computational costs (if asked)
- Understand comparison to existing tools
- Can explain integration complexity

### **If Running Short:**
- Skip detailed feature explanations
- Compress architecture overview
- Focus on results and impact

### **If Running Long:**
- Cut future work section
- Simplify feature engineering explanation
- Move faster through technical architecture

---

## **KEY PHRASES TO MEMORIZE**

1. **Opening:** *"99.55% precision while processing over 17 million name occurrences"*
2. **Problem:** *"Manual curation at this scale simply cannot work"*
3. **Innovation:** *"Five critical features that capture different aspects of entity similarity"*
4. **Results:** *"Only 45 false positives out of 10,000 positive predictions"*
5. **Impact:** *"99.23% reduction in required comparisons"*
6. **Vision:** *"Enabling researchers to discover connections that were previously invisible"*
7. **Closing:** *"From 4.8 million name variants to unified scholarly identity"*

These phrases anchor your talk and can be delivered with confidence even if you lose your place in the script.
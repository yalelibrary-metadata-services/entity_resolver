## Understanding Distance Metrics for Text Embeddings

When working with word embeddings, we need ways to measure how similar or different two vectors are. Here are the three most important metrics:

### Cosine Similarity
**What it measures**: The angle between two vectors, ignoring their magnitudes.  
**Range**: -1 (completely opposite) to +1 (identical direction)  
**Intuition**: Think of two arrows pointing in space—cosine similarity tells you if they point in the same direction, regardless of their length.  
**Why it's preferred for embeddings**: Word vectors can have different magnitudes due to word frequency, but we care about semantic direction. "Cat" and "kitten" should be similar even if one vector is longer.  
**Formula**: cos(θ) = (A · B) / (|A| × |B|)

### Euclidean Distance
**What it measures**: The straight-line distance between two points in space.  
**Range**: 0 (identical) to ∞ (very different)  
**Intuition**: If vectors were cities on a map, this would be the "as the crow flies" distance between them.  
**When to use**: Better for data where magnitude matters, like measuring physical distances or when vectors are normalized.  
**Formula**: √[(A₁-B₁)² + (A₂-B₂)² + ... + (Aₙ-Bₙ)²]

### Dot Product
**What it measures**: The product of vector magnitudes times the cosine of the angle between them.  
**Range**: -∞ to +∞  
**Intuition**: Combines both the direction similarity (like cosine) and the magnitude information. Longer vectors in the same direction give higher scores.  
**When to use**: Useful when both direction and magnitude matter, or as a component in other calculations.  
**Formula**: A · B = A₁×B₁ + A₂×B₂ + ... + Aₙ×Bₙ

### Key Insight for Entity Resolution
For word embeddings, **cosine similarity is typically preferred** because it focuses on semantic direction rather than word frequency effects. Two words with similar meanings should be considered similar regardless of how often they appear in training data (which affects vector magnitude).
I need to prepare content for an AI workshop with Yale graduate students from a wide variety of backgrounds (STEM to humanities). My goal is to present for 45 minutes on text embeddings and classification:

Learning Objectives:

- Understand how text embeddings encode semantic meaning
- Apply embeddings to entity resolution challenges
- Implement classification with minimal labeled data

Topics Covered in Workshop:

- Evolution from Word2Vec to modern text embeddings
- OpenAI text-embedding-3-small for large-scale applications
- Hot-deck imputation with embeddings
- Mistral AI Classifier Factory introduction
- Vector databases (Weaviate) for similarity search

Provide an integrated, detailed breakdown of key concepts and prepare robust, complete content for my presentation.

Your task is to produce a well-formed .ipynb code notebook for hosting on Google Colab that gives a realistic demo of my entity resolution pipeline.

The notebook should be pedagogically oriented. Concepts and code should be broken down and spoon fed: ideally, one unit per cell, without complex function definitions.

The notebook should demo the classification of names in the "Schubert, Franz" cluster (e.g., composer vs. artist), with proper Weaviate integration.

It should also demonstrate Weaviate query functionality: e.g., use `composite` strings of records with null `subjects` to retrieve subjects from similar records using `near_vector` search.
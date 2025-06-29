Review and analyze my entity resolution pipeline (directly: `/Users/tt434/Dropbox/YUL/2025/msu/tmp/entity_resolver`) and think deeply to analyze its functionality and results.

Review the project code to refactor and rewrite the `project_structure.md` and `README.md` files to ensure accuracy, readability, and completeness. 

Pipeline results and visualizations are in `data/output`.

- **Use your Analysis tool to analyze the test result CSV included in the project knowledge.**
- Focus on analyzing output and functionality to explain how the pipeline works at each stage.
- Prepare explanatory analysis with diagrams and visualizations as appropriate.
- Use MCP filesystem tools to read and, if necessary, modify files on disk.
- Access only those files that are required for understanding how to perform the work. Do not hunt for extraneous information.


## Dataset Structure
### Fields in the Source Data (derived from MARC 21 records in the Yale Library catalog)
_Files are encoded as CSV, with commas as field separators._
- **composite**: Composite text containing non-null values from all other fields (always present).
- **person**: Extracted personal name (always present).
- **roles**: Relationship of the person to the work (always present; typically defaults to "Contributor" or "Subject").
- **title**: Title of the work (always present).
- **provision**: Publication details (place, date).
- **subjects**: Subject classifications.
- **personId**: Unique identifier for each unresolved person entity (note that the key `id` is a reserved value in Weaviate).
- **setfit_prediction**: Category from `data/input/revised_taxonomy_final.json` that indicates the domain of activity associated with the person via the composite string.
- **is_parent_category**: Indicator for the level in the taxonomy.

## Ground Truth Data
- **Match pairs format** (comma separated):
  ```
  left,right,match
  16044091#Agent700-32,9356808#Agent100-11,true
  16044091#Agent700-32,9940747#Hub240-13-Agent,true
  ```
- **Dataset record format** (comma separated):
```
composite,person,roles,title,provision,subjects,personId
"Contributor: Allen, William
Title: Dēmosthenous Logoi dēmēgorikoi dōdeka: Demosthenis Orationes de republica duodecim
Variant titles: Logoi dēmēgorikoi dōdeka; Orationes de republica duodecim
Attribution: cum Wolfiana interpretatione ; accessit Philippi epistola, a Gulielmo Allen, A.M.
Provision information: Oxonii [Oxford, England]: Typis et sumtu N. Bliss, 1810; Ed. nova.","Allen, William",Contributor,Dēmosthenous Logoi dēmēgorikoi dōdeka: Demosthenis Orationes de republica duodecim,"Oxonii [Oxford, England]: Typis et sumtu N. Bliss, 1810 Ed. nova.",,2117946#Agent700-25,Humanities, Thought, and Interpretation,TRUE
```

## Features
Currently enabled features and parameters in the classifier model:
enabled:  
    - person_cosine
    - person_title_squared
    - composite_cosine
    - taxonomy_dissimilarity
    - birth_death_match    
  
Of course. Here is the integrated English text, synthesized from your notes while preserving the original meaning.

1. Background and Aims
Human cognition processes high-dimensional and diverse information from the external world, condensing it into a lower-dimensional structure with a high degree of abstraction to form a categorical and hierarchical knowledge system. Knowledge represented through language is also thought to be formed through this process of abstraction and categorization. The abstract and hierarchical characteristics observed in linguistic knowledge are a reflection of these underlying human information processing mechanisms.

The purpose of this research is to uncover the low-dimensional latent structures inherent in this cognitive process by analyzing the linguistic knowledge in which these structures are reflected. We hypothesize that the consistent formation of categorical and hierarchical knowledge across various domains implies a strong constraint or physiological basis in our cognitive functions, compelling an aggregation of information through high-level abstractions.

2. Theoretical Framework
To model this cognitive process, we employ CEBRA (Contrastive Embedding Based on Relevant Auxiliary variables), a contrastive learning framework originally developed for neural data analysis with a theoretical foundation in non-linear Independent Component Analysis (ICA).

CEBRA operates on the assumption that a smooth, bijective (one-to-one) mapping exists between the unobserved latent space and the high-dimensional observation space (e.g., text embeddings). Its key strength lies in its theoretical guarantees: under the condition that latent variables are sufficiently diverse given an auxiliary variable (e.g., data labels), the learned embedding space is guaranteed to be topologically homeomorphic to the true latent space. This ensures that the fundamental structure of the data is faithfully preserved.

Furthermore, this process yields consistent embeddings, meaning that repeated training runs produce similar geometric structures. Unlike autoencoder-based models such as VAEs, CEBRA does not use a decoder. This architectural choice avoids the bottleneck of reconstruction and contributes to producing more consistent and reliable representations of the latent structure. It is therefore a suitable method for modeling the consistent abstraction and categorization functions of human cognition.

3. Experimental Design
Data and Preprocessing
We adopted text embeddings as an experimental proxy for the knowledge space that indirectly reflects the complex information processing occurring in the human brain. While embeddings from models like BERT exist in a high-dimensional space (e.g., 768 dimensions), it is known that their intrinsic dimensionality is much lower, with most of the variance concentrated in approximately 12-15 dimensions. However, these dimensions are not inherently interpretable.

Our experiments utilized two main datasets: dair-ai/emotion and GoEmotions.

Methods
We used the emotion labels associated with the text data as auxiliary variables for CEBRA. The objective was to embed the high-dimensional textual information into an interpretable, low-dimensional latent space.

For the dair-ai/emotion dataset, we tested combinations of:

Four different text embedding models (all-MiniLM-L6-v2, etc.).

Output dimensions ranging from 2 to 12.

For each configuration, we trained the model 10 times to evaluate the consistency of the resulting embeddings.

For the GoEmotions dataset, we tested:

Four different text embedding models.

Output dimensions ranging from 2 to 8 and 24 to 29 (corresponding to the number of emotion labels).

Consistency evaluation was performed only for the BERT embeddings.


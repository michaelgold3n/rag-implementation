reranker explanation:
- top_n=5: Limits the reranking to the top 5 results.
- model="BAAI/bge-reranker-large": Specifies the model used for reranking.

so the reranker  is supposed to refine the results by selecting the top 5 outputs based on embedding similarity.

## requirements: 


1) might have to install a bunch of dependencies if streamlit gives you other errors that are unrelated to the reranker. just do it piece by piece
2) then make a .env file with ur api keys from:

OPENAI_API_KEY="xyz"

  LLAMA_CLOUD_API_KEY="xyz"

### P.S also, the ideal thing we tried to do was actually a multi-modal rag like in this repo: https://github.com/langchain-ai/langchain/blob/master/cookbook/Multi_modal_RAG.ipynb.

- failed pretty bad at that and hoping the main reason this rag isn't working is just the reranker and not a lack of multimodality in order to process the documents in the repo (bc our .py file is using OAI api, which only processes text, rather than Gemini 1.5, which does images...but the documentation was horrible for google rag stuff)
- you can find one of our attempts in doing that in mm1.py, but its not very interesting and pretty messy. at this point we've accepted that multimodality isnt something we care too much about. mostly just rag that works.

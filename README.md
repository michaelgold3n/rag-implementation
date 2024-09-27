## reranker explanation:
- top_n=5: Limits the reranking to the top 5 results.
- model="BAAI/bge-reranker-large": Specifies the model used for reranking.

so the reranker  is supposed to refine the results by selecting the top 5 outputs based on embedding similarity.

## requirements: 


1) might have to install a bunch of dependencies if streamlit gives you other errors that are unrelated to the reranker. just do it piece by piece
2) then make a .env file with ur api keys from:

OPENAI_API_KEY="xyz"

  LLAMA_CLOUD_API_KEY="xyz"

## btw,

### ideal behavior / best results can be foud this repo: https://github.com/langchain-ai/langchain/blob/master/cookbook/Multi_modal_RAG.ipynb .

- unfortunately there were limitations when attempting to reimplement that and hoping the main reason that demo.py isn't working is just the reranker and not a lack of multimodality in order to process the documents in the repo (bc the .py file is using OAI api, which only processes text, rather than Gemini 1.5, which does text+images...but the documentation wasn't ideal for google rag stuff)
- you can find an attempt of this in mm1.py, but its not very interesting and pretty messy. at this point accepted that multimodality isn't the highest priority. rag that works exceptionally well is more important
- and actually, after figuring out making the rag better, i might just try gpt-4 for multimodal now that i think of it. only tried gemini bc it's free but it turns out to be very complex and doesn't have the best documentation.

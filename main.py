from gensim.models import Word2Vec
from IPython.display import display, clear_output
from fastapi import FastAPI, HTTPException
import uvicorn

app = FastAPI()

model1 = Word2Vec.load("word2vec_model1_2016.model")
model2 = Word2Vec.load("word2vec_model_2024.model")


def get_similar_words(model, word):
    try:
        similar = model.wv.most_similar(word, topn=10)
        return similar
    except KeyError:
        return f"'{word}' not found in the vocabulary."


@app.get("/similar/{model_id}/{word}")
async def find_similar_words(model_id: int, word: str):
    if model_id == 1:
        result = get_similar_words(model1, word)
    elif model_id == 2:
        result = get_similar_words(model2, word)
    else:
        raise HTTPException(status_code=404, detail="Model not found")

    if result is None:
        raise HTTPException(
            status_code=404, detail=f"'{word}' not found in the vocabulary."
        )

    return {"word": word, "similar_words": result}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
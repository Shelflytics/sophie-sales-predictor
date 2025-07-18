from fastapi import FastAPI

app = FastAPI()

# Commands to run
# 1. pip install "fastapi[standard]" or conda install "fastapi[standard]"
# 2. fastapi dev main.py
# You will be able to find your docs at https://127.0.0.1/docs/

@app.get("/api/meow")
def meow(request):
    data = request.json()
    postal_code = data["postal_code"]
    return {"response": "Hello!"}
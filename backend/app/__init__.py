from fastapi import FastAPI

from app.routers import router  # Import the main router from the routers package

app = FastAPI()

# Include routers
app.include_router(router)  # Use the main router from routers/__init__.py which has all sub-routers included


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Welcome to the FastAPI application!"}

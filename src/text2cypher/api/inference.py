from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ValidationError, Field, validator
from loguru import logger
import os
from pathlib import Path
import torch
from typing import Optional

from text2cypher.api.config import settings
from text2cypher.finetuning.models.llama_model import LlamaText2CypherModel
from text2cypher.finetuning.utils.logger import setup_logger
from text2cypher.finetuning.utils.tokenization_utils import normalize_cypher_query

# Initialize model variable at module level
model = None

app = FastAPI(
    title="Text2Cypher Generator API",
    description="API for generating Cypher queries from natural language text",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Setup logging
setup_logger()

# Initialize model at startup
@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    try:
        global model
        model_path = os.getenv("MODEL_PATH", "/app/models/hf_model")
        logger.info(f"Loading model from: {model_path}")
        
        # Try loading from Hugging Face format first, then fallback to checkpoint
        if os.path.exists(model_path) and os.path.isdir(model_path):
            model = LlamaText2CypherModel.from_pretrained(model_path)
        else:
            # Fallback to checkpoint loading
            model = LlamaText2CypherModel.load_model_from_checkpoint(
                checkpoint_path=str(model_path),
            )
        
        model.setup_inference()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError("Failed to initialize model")

@app.get("/")
async def root():
    return {
        "message": "Text2Cypher Generator API",
        "version": "1.0.0",
        "endpoints": ["/health", "/generate_cypher", "/docs"]
    }

class Text2CypherRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural language question to convert to Cypher query")
    schema: Optional[str] = Field(None, description="Graph database schema (nodes, relationships, properties)")
    max_length: int = Field(default=512, ge=1, le=1024, description="Maximum length of generated Cypher query")

    @validator('question')
    def clean_question_input(cls, v):
        logger.info("Validating question input")
        try:
            v = v.replace('\n', ' ').replace('\r', ' ')
            v = ' '.join(v.split())
            logger.info("Question cleaned in validator")
            return v
        except Exception as e:
            logger.error(f"Error in question validator: {str(e)}")
            raise ValueError(f"Invalid question format: {str(e)}")

    @validator('schema')
    def clean_schema_input(cls, v):
        if v is None:
            return v
        logger.info("Validating schema input")
        try:
            # Keep schema formatting more intact since it's structured
            v = v.strip()
            logger.info("Schema cleaned in validator")
            return v
        except Exception as e:
            logger.error(f"Error in schema validator: {str(e)}")
            raise ValueError(f"Invalid schema format: {str(e)}")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Find all patients with diabetes",
                "schema": "Node properties: - **Patient** - `name`: STRING - **Condition** - `name`: STRING Relationships: (:Patient)-[:HAS_CONDITION]->(:Condition)",
                "max_length": 256
            }
        }

class CypherResponse(BaseModel):
    cypher_query: str

    class Config:
        schema_extra = {
            "example": {
                "cypher_query": "MATCH (p:Patient)-[:HAS_CONDITION]->(c:Condition {name: 'diabetes'}) RETURN p"
            }
        }

@app.post("/generate_cypher", response_model=CypherResponse)
async def generate_cypher(request: Text2CypherRequest):
    logger.info("Incoming request to /generate_cypher endpoint")
    logger.debug("Raw request received")
    try:
        logger.info("Received generate_cypher request")
        logger.debug(f"Original request: {request.dict()}")
        if not request.question:
            raise HTTPException(status_code=400, detail="Empty question")
        
        logger.info("Generating Cypher query...")
        cypher_query = model.generate_cypher(
            question=request.question, 
            schema=request.schema, 
            max_length=request.max_length
        )
        
        # Normalize the generated Cypher query
        normalized_query = normalize_cypher_query(cypher_query)
        
        logger.info("Cypher generation successful")
        return CypherResponse(cypher_query=normalized_query)
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating Cypher: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint with model status."""
    try:
        global model
        model_loaded = model is not None and hasattr(model, 'is_loaded') and model.is_loaded()
        
        return {
            "status": "healthy" if model_loaded else "unhealthy",
            "model_loaded": model_loaded,
            "timestamp": logger._core.start_time.isoformat() if hasattr(logger, '_core') else None
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": str(e)
        }

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming {request.method} request to {request.url}")
    try:
        body = await request.body()
        if body:
            logger.debug(f"Request body: {body.decode()}")
    except Exception as e:
        logger.error(f"Could not log request body: {str(e)}")
    response = await call_next(request)
    return response

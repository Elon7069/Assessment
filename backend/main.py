# main.py
# Advanced FastAPI backend for pipeline DAG validation
# Features: NetworkX integration, Pydantic models, CORS support, comprehensive error handling

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any
import networkx as nx
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VectorShift Pipeline API",
    description="Backend API for pipeline DAG validation and analysis",
    version="1.0.0"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class NodeData(BaseModel):
    """Individual node in the pipeline"""
    id: str = Field(..., description="Unique node identifier")
    type: str = Field(..., description="Node type (e.g., 'input', 'output', 'llm')")
    data: Dict[str, Any] = Field(default={}, description="Additional node data")
    
    @validator('id')
    def validate_id(cls, v):
        if not v or not v.strip():
            raise ValueError('Node ID cannot be empty')
        return v.strip()
    
    @validator('type')
    def validate_type(cls, v):
        if not v or not v.strip():
            raise ValueError('Node type cannot be empty')
        return v.strip()

class EdgeData(BaseModel):
    """Connection between nodes in the pipeline"""
    id: str = Field(..., description="Unique edge identifier")
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    sourceHandle: str = Field(default="", description="Source handle ID")
    targetHandle: str = Field(default="", description="Target handle ID")
    
    @validator('source', 'target')
    def validate_node_ids(cls, v):
        if not v or not v.strip():
            raise ValueError('Source and target node IDs cannot be empty')
        return v.strip()
    
    @validator('id')
    def validate_edge_id(cls, v):
        if not v or not v.strip():
            raise ValueError('Edge ID cannot be empty')
        return v.strip()

class PipelineRequest(BaseModel):
    """Complete pipeline data for validation"""
    nodes: List[NodeData] = Field(..., description="List of pipeline nodes")
    edges: List[EdgeData] = Field(..., description="List of pipeline edges")
    
    @validator('nodes')
    def validate_nodes(cls, v):
        if not v:
            return v  # Allow empty pipelines
        
        # Check for duplicate node IDs
        node_ids = [node.id for node in v]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError('Duplicate node IDs found')
        return v
    
    @validator('edges')
    def validate_edges(cls, v, values):
        if not v:
            return v  # Allow pipelines with no edges
        
        # Check for duplicate edge IDs
        edge_ids = [edge.id for edge in v]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError('Duplicate edge IDs found')
        
        # Validate that edge references exist in nodes
        if 'nodes' in values and values['nodes']:
            node_ids = {node.id for node in values['nodes']}
            for edge in v:
                if edge.source not in node_ids:
                    raise ValueError(f'Edge references non-existent source node: {edge.source}')
                if edge.target not in node_ids:
                    raise ValueError(f'Edge references non-existent target node: {edge.target}')
        
        return v

class PipelineResponse(BaseModel):
    """Response from pipeline analysis"""
    num_nodes: int = Field(..., description="Number of nodes in the pipeline")
    num_edges: int = Field(..., description="Number of edges in the pipeline")
    is_dag: bool = Field(..., description="Whether the pipeline forms a valid DAG")
    cycles: List[List[str]] = Field(default=[], description="List of detected cycles (if any)")
    analysis: Dict[str, Any] = Field(default={}, description="Additional graph analysis")
    message: str = Field(..., description="Human-readable status message")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    version: str

# Graph analysis utilities
class GraphAnalyzer:
    """Advanced graph analysis using NetworkX"""
    
    @staticmethod
    def build_graph(nodes: List[NodeData], edges: List[EdgeData]) -> nx.DiGraph:
        """Build NetworkX directed graph from pipeline data"""
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for node in nodes:
            G.add_node(node.id, type=node.type, data=node.data)
        
        # Add edges
        for edge in edges:
            G.add_edge(
                edge.source, 
                edge.target,
                edge_id=edge.id,
                sourceHandle=edge.sourceHandle,
                targetHandle=edge.targetHandle
            )
        
        return G
    
    @staticmethod
    def analyze_dag(G: nx.DiGraph) -> Dict[str, Any]:
        """Comprehensive DAG analysis"""
        analysis = {
            'is_dag': nx.is_directed_acyclic_graph(G),
            'cycles': [],
            'strongly_connected_components': [],
            'topological_sort': [],
            'isolated_nodes': [],
            'graph_stats': {
                'density': 0.0,
                'number_of_selfloops': 0,
                'number_of_edges': G.number_of_edges(),
                'number_of_nodes': G.number_of_nodes()
            }
        }
        
        try:
            # Find cycles if graph is not a DAG
            if not analysis['is_dag']:
                try:
                    # Find all simple cycles
                    cycles = list(nx.simple_cycles(G))
                    analysis['cycles'] = cycles[:10]  # Limit to first 10 cycles
                except Exception as e:
                    logger.warning(f"Error finding cycles: {e}")
                    analysis['cycles'] = []
            
            # Get strongly connected components
            analysis['strongly_connected_components'] = [
                list(component) for component in nx.strongly_connected_components(G)
                if len(component) > 1  # Only multi-node components
            ]
            
            # Topological sort (if DAG)
            if analysis['is_dag'] and G.number_of_nodes() > 0:
                analysis['topological_sort'] = list(nx.topological_sort(G))
            
            # Find isolated nodes
            analysis['isolated_nodes'] = list(nx.isolates(G))
            
            # Graph statistics
            if G.number_of_nodes() > 0:
                analysis['graph_stats']['density'] = nx.density(G)
                analysis['graph_stats']['number_of_selfloops'] = nx.number_of_selfloops(G)
            
        except Exception as e:
            logger.error(f"Error in graph analysis: {e}")
            analysis['error'] = str(e)
        
        return analysis

# API Routes
@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="VectorShift Pipeline API is running",
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def detailed_health():
    """Detailed health check with dependencies"""
    try:
        # Test NetworkX functionality
        test_graph = nx.DiGraph()
        test_graph.add_edge("A", "B")
        is_working = nx.is_directed_acyclic_graph(test_graph)
        
        return HealthResponse(
            status="healthy" if is_working else "degraded",
            message="All systems operational" if is_working else "NetworkX issues detected",
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.post("/pipelines/parse", response_model=PipelineResponse)
async def parse_pipeline(pipeline: PipelineRequest):
    """
    Analyze pipeline and validate DAG structure
    
    This endpoint:
    1. Validates input data structure
    2. Builds NetworkX directed graph
    3. Performs cycle detection
    4. Returns comprehensive analysis
    """
    try:
        logger.info(f"Analyzing pipeline with {len(pipeline.nodes)} nodes and {len(pipeline.edges)} edges")
        
        # Build graph using NetworkX
        graph = GraphAnalyzer.build_graph(pipeline.nodes, pipeline.edges)
        
        # Perform comprehensive analysis
        analysis = GraphAnalyzer.analyze_dag(graph)
        
        # Generate human-readable message
        if analysis['is_dag']:
            if len(pipeline.nodes) == 0:
                message = "Empty pipeline - no nodes to analyze"
            elif len(pipeline.edges) == 0:
                message = f"Pipeline with {len(pipeline.nodes)} isolated nodes - valid but disconnected"
            else:
                message = f"✅ Valid DAG with {len(pipeline.nodes)} nodes and {len(pipeline.edges)} edges"
        else:
            cycle_count = len(analysis['cycles'])
            message = f"❌ Invalid DAG - contains {cycle_count} cycle(s). Please remove circular dependencies."
        
        response = PipelineResponse(
            num_nodes=len(pipeline.nodes),
            num_edges=len(pipeline.edges),
            is_dag=analysis['is_dag'],
            cycles=analysis['cycles'],
            analysis=analysis,
            message=message
        )
        
        logger.info(f"Analysis complete: is_dag={analysis['is_dag']}")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in pipeline analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during pipeline analysis")

@app.get("/pipelines/example")
async def get_example_pipeline():
    """Get an example valid pipeline for testing"""
    example = {
        "nodes": [
            {"id": "input-1", "type": "input", "data": {"label": "User Input"}},
            {"id": "llm-1", "type": "llm", "data": {"model": "gpt-3.5-turbo"}},
            {"id": "output-1", "type": "output", "data": {"format": "text"}}
        ],
        "edges": [
            {"id": "e1", "source": "input-1", "target": "llm-1"},
            {"id": "e2", "source": "llm-1", "target": "output-1"}
        ]
    }
    return example

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return HTTPException(status_code=400, detail=str(exc))

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

import asyncio
import sys
import os

# Force utf-8 output to avoid Windows console errors
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure the root directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.tools import RAGTool

async def main():
    print("Initializing RAGTool...")
    rag = RAGTool()
    
    query = "What does this business sell? Products, target audience, selling points, brand personality, pricing."
    print(f"\nQuerying RAG with: '{query}'\n")
    print("="*60)
    
    try:
        result = await rag.execute(query=query, user_id="test_user")
        
        with open("rag_test_output.txt", "w", encoding="utf-8") as f:
            if result.get("success"):
                f.write("RETRIEVAL SUCCESSFUL\n\n")
                f.write(f"Number of chunks retrieved: {result.get('chunks_count', 0)}\n")
                
                distances = result.get('distances', [])
                if distances:
                    f.write(f"Distances: min={min(distances):.4f}, max={max(distances):.4f}, avg={sum(distances)/len(distances):.4f}\n")
                
                f.write("\n" + "="*60 + "\n")
                f.write("RAW RETRIEVED TEXT:\n")
                f.write("="*60 + "\n")
                f.write(result.get("retrieved", "No content retrieved") + "\n")
                f.write("="*60 + "\n")
                
                # Print individual chunks
                chunks = result.get("chunks", [])
                for i, chunk in enumerate(chunks):
                    f.write(f"\n--- CHUNK {i+1} ---\n")
                    if isinstance(chunk, dict):
                        doc = chunk.get("document", "No document")
                        meta = chunk.get("metadata", {})
                        f.write(f"Content: {doc}\n")
                        if meta:
                            f.write(f"Metadata: {meta}\n")
            else:
                f.write(f"RETRIEVAL FAILED: {result.get('error')}\n")
            
    except Exception as e:
        with open("rag_test_output.txt", "w", encoding="utf-8") as f:
            f.write(f"ERROR DURING EXECUTION: {e}\n")

if __name__ == "__main__":
    # Workaround for Windows asyncio loop
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

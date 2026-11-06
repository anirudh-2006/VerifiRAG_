# src/pipeline.py
from dotenv import load_dotenv
load_dotenv()

from src.rag_chain import ask_structured
from src.verifier import verify_derivation, print_verification_report

def run(question: str):
    print(f"\nQuestion: {question}")
    print("\n[1] Running RAG + Concept Graph retrieval...")

    structured = ask_structured(question)

    if structured is None:
        print("Failed to get structured output from LLM. Try rephrasing the question.")
        return

    print(f"\n[2] LLM produced {len(structured.get('steps', []))} steps")
    print(f"    Concept: {structured.get('concept')}")
    print(f"    Final result: {structured.get('final_result')}")

    print("\n[3] Running symbolic verification...")
    results = verify_derivation(structured)

    print_verification_report(structured, results)

if __name__ == "__main__":
    run("Derive the Fourier transform of a rectangular pulse signal.")
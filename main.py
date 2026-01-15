from src.pipeline import RAGPipeline


if __name__ == "__main__":

    pipeline = RAGPipeline(
        data_paths=["./data"],
        persist_dir="./chroma_db",

        chunking_mode="recursive",
        enable_rerank=True,
        enable_compression=True,
        top_k=5,
        verbose=True,
    )


    pipeline.build_index(rebuild=True)
    pipeline.load_models()

    print("\nPipeline ready. Ask questions.\n")

    while True:
        query = input("Ask (or 'exit'): ").strip()
        if query.lower() == "exit":
            break

        answer = pipeline.run(query)
        print("\n--- ANSWER ---\n")
        print(answer)

from rag import retrieve_docs

query = "What are the pricing plans available?"

result = retrieve_docs(query)

print(result)
Questions: 36 | chunk_size=900 overlap=150

| Chunking | Chunks | Retriever | Hit@1 | Hit@3 | Hit@5 | MRR@10 | ms/query |
|---|---:|---|---:|---:|---:|---:|---:|
| fixed | 54 | bm25 | 0.722 | 0.833 | 0.861 | 0.785 | 0.3 |
| lines | 51 | bm25 | 0.583 | 0.778 | 0.833 | 0.706 | 0.3 |

- `fixed` / bm25: not retrieved in top 10: ['q29', 'q33']
- `lines` / bm25: not retrieved in top 10: ['q22', 'q29']

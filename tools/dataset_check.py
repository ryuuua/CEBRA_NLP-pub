from datasets import load_dataset
from itertools import islice


dataset = load_dataset(
    "institutional/institutional-books-1.0",
    split="train",
    streaming=True
)

for i, row in enumerate(dataset):
    if row["text_by_page_gen"] and any(p.strip() for p in row["text_by_page_gen"]):
        print("=== Found non-empty book ===")
        print("title:", row["title_src"])
        print("author:", row["author_src"])
        print("year:", row["date1_src"])
        # 最初に見つけた本文ありのページを表示
        for idx, page in enumerate(row["text_by_page_gen"]):
            if page.strip():
                print(f"--- page {idx+1} sample ---")
                print(page[:500])
                break
        break
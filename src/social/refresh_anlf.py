from src.social.anlf_collector import (
    collect_anlf_snapshot,
    merge_snapshot_into_master,
    save_raw_snapshot,
)

from src.social.anlf_relevance_gate import (
    classify_all_posts,
    load_master_posts,
    write_relevant_dataset,
    write_report,
)

from src.social.build_anlf_index import (
    build_chunks,
    build_embeddings,
    load_posts,
    save_chunks,
)


def main():

    print(
        "=" * 60
    )

    print(
        "ANLF DAILY REFRESH"
    )

    print(
        "=" * 60
    )

    # ==================================================
    # STEP 1 — COLLECT
    # ==================================================

    print(
        "\n[1/3] Collecting posts + comments..."
    )

    snapshot = (
        collect_anlf_snapshot()
    )

    raw_path = (
        save_raw_snapshot(
            snapshot
        )
    )

    (
        new_posts,
        new_comments,
    ) = merge_snapshot_into_master(
        snapshot
    )

    print(
        f"New posts: {new_posts}"
    )

    print(
        f"New comments: {new_comments}"
    )

    print(
        f"Snapshot: {raw_path}"
    )

    # ==================================================
    # STEP 2 — RELEVANCE GATE
    # ==================================================

    print(
        "\n[2/3] Running relevance gate..."
    )

    master_posts = (
        load_master_posts()
    )

    classifications = (
        classify_all_posts(
            master_posts
        )
    )

    (
        relevant_count,
        excluded_count,
    ) = write_relevant_dataset(
        master_posts,
        classifications,
    )

    write_report(
        master_posts,
        classifications,
    )

    print(
        f"Relevant posts: {relevant_count}"
    )

    print(
        f"Excluded posts: {excluded_count}"
    )

    # ==================================================
    # STEP 3 — REBUILD SOCIAL INDEX
    # ==================================================

    print(
        "\n[3/3] Rebuilding ANLF index..."
    )

    relevant_posts = (
        load_posts()
    )

    chunks = (
        build_chunks(
            relevant_posts
        )
    )

    save_chunks(
        chunks
    )

    embeddings = (
        build_embeddings(
            chunks
        )
    )

    admin_chunks = sum(
        1
        for chunk in chunks
        if chunk.get(
            "chunk_type"
        ) == "ADMIN_INFO"
    )

    comment_chunks = sum(
        1
        for chunk in chunks
        if chunk.get(
            "chunk_type"
        ) == "COMMENT"
    )

    print()

    print(
        "=" * 60
    )

    print(
        "ANLF REFRESH COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"New posts:        {new_posts}"
    )

    print(
        f"New comments:     {new_comments}"
    )

    print(
        f"Relevant posts:   {relevant_count}"
    )

    print(
        f"Excluded posts:   {excluded_count}"
    )

    print(
        f"Admin chunks:     {admin_chunks}"
    )

    print(
        f"Comment chunks:   {comment_chunks}"
    )

    print(
        f"Total chunks:     {len(chunks)}"
    )

    print(
        f"Embedding shape:  {embeddings.shape}"
    )


if __name__ == "__main__":

    main()
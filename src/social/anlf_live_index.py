import threading
from pathlib import Path

import numpy as np

from src.social.anlf_retrieval import (
    CHUNKS_FILE,
    EMBEDDINGS_FILE,
    load_chunks,
)


class ANLFLiveIndex:
    """
    Thread-safe live loader for the ANLF social index.

    The API keeps the last known-good index in memory.

    When the daily refresh pipeline rebuilds the files,
    the next request detects the file change and reloads
    the new index without requiring a FastAPI restart.
    """

    def __init__(self):

        self._lock = threading.Lock()

        self.chunks = None
        self.embeddings = None

        self.signature = None


    # ==================================================
    # FILE SIGNATURE
    # ==================================================

    def _file_signature(
        self,
    ):

        paths = [
            Path(
                CHUNKS_FILE
            ),
            Path(
                EMBEDDINGS_FILE
            ),
        ]

        if not all(
            path.exists()
            for path in paths
        ):

            return None

        return tuple(
            (
                path.stat().st_mtime_ns,
                path.stat().st_size,
            )
            for path in paths
        )


    # ==================================================
    # LOAD FROM DISK
    # ==================================================

    def _load_from_disk(
        self,
    ):

        chunks = (
            load_chunks()
        )

        embeddings = (
            np.load(
                EMBEDDINGS_FILE
            )
        )

        if len(
            chunks
        ) != len(
            embeddings
        ):

            raise RuntimeError(
                "ANLF chunks and embeddings "
                "are out of sync."
            )

        return (
            chunks,
            embeddings,
        )


    # ==================================================
    # INITIAL LOAD
    # ==================================================

    def load_initial(
        self,
    ) -> bool:

        try:

            signature = (
                self._file_signature()
            )

            if signature is None:

                return False

            (
                chunks,
                embeddings,
            ) = self._load_from_disk()

            with self._lock:

                self.chunks = (
                    chunks
                )

                self.embeddings = (
                    embeddings
                )

                self.signature = (
                    signature
                )

            return True

        except Exception:

            return False


    # ==================================================
    # LIVE RELOAD
    # ==================================================

    def reload_if_changed(
        self,
    ) -> str:
        """
        Possible return values:

        unchanged
        reloaded
        unavailable
        reload_failed
        """

        current_signature = (
            self._file_signature()
        )

        if current_signature is None:

            return "unavailable"

        if (
            current_signature
            == self.signature
        ):

            return "unchanged"

        with self._lock:

            # Another request may already have
            # reloaded the index while we waited
            # for the lock.

            current_signature = (
                self._file_signature()
            )

            if current_signature is None:

                return "unavailable"

            if (
                current_signature
                == self.signature
            ):

                return "unchanged"

            try:

                (
                    new_chunks,
                    new_embeddings,
                ) = self._load_from_disk()

            except Exception:

                # Keep serving the previous
                # known-good in-memory index.
                return "reload_failed"

            # Swap only after both files were
            # successfully loaded and validated.

            self.chunks = (
                new_chunks
            )

            self.embeddings = (
                new_embeddings
            )

            self.signature = (
                current_signature
            )

            return "reloaded"


    # ==================================================
    # SNAPSHOT
    # ==================================================

    def get_snapshot(
        self,
    ):

        with self._lock:

            return (
                self.chunks,
                self.embeddings,
            )


    # ==================================================
    # STATUS
    # ==================================================

    def is_loaded(
        self,
    ) -> bool:

        (
            chunks,
            embeddings,
        ) = self.get_snapshot()

        return (
            chunks is not None
            and embeddings is not None
        )


    def chunk_count(
        self,
    ) -> int:

        (
            chunks,
            _,
        ) = self.get_snapshot()

        if chunks is None:

            return 0

        return len(
            chunks
        )
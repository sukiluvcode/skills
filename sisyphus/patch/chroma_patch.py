# -*- coding:utf-8 -*-
"""
@File    :   chroma_patch.py
@Time    :   2024/05/04 18:20:45
@Author  :   soike
@Version :   1.0
@Contact :   luvusoike@icloud.com
@License :   MIT Lisence
@Desc    :   patch chroma db
"""

import asyncio
import uuid
from typing import Iterable, Optional, Any, List

from langchain_community.vectorstores.chroma import Chroma


class AsyncChroma(Chroma):
    """implement aadd_texts method of `Chroma`"""

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]], optional): Optional list of IDs.

        Returns:
            List[str]: List of IDs of the added texts.
        """
        # TODO: Handle the case where the user doesn't provide ids on the Collection
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        embeddings = None
        texts = list(texts)
        if self._embedding_function is not None:
            embeddings = await self._embedding_function.aembed_documents(texts)
        if metadatas:
            # fill metadatas with empty dicts if somebody
            # did not specify metadata for all texts
            length_diff = len(texts) - len(metadatas)
            if length_diff:
                metadatas = metadatas + [{}] * length_diff
            empty_ids = []
            non_empty_ids = []
            for idx, m in enumerate(metadatas):
                if m:
                    non_empty_ids.append(idx)
                else:
                    empty_ids.append(idx)
            if non_empty_ids:
                metadatas = [metadatas[idx] for idx in non_empty_ids]
                texts_with_metadatas = [texts[idx] for idx in non_empty_ids]
                embeddings_with_metadatas = (
                    [embeddings[idx] for idx in non_empty_ids]
                    if embeddings
                    else None
                )
                ids_with_metadata = [ids[idx] for idx in non_empty_ids]
                try:
                    await asyncio.to_thread(
                        self._collection.upsert,
                        metadatas=metadatas,
                        embeddings=embeddings_with_metadatas,
                        documents=texts_with_metadatas,
                        ids=ids_with_metadata,
                    )
                except ValueError as e:
                    if 'Expected metadata value to be' in str(e):
                        msg = (
                            'Try filtering complex metadata from the document using '
                            'langchain_community.vectorstores.utils.filter_complex_metadata.'
                        )
                        raise ValueError(e.args[0] + '\n\n' + msg)
                    else:
                        raise e
            if empty_ids:
                texts_without_metadatas = [texts[j] for j in empty_ids]
                embeddings_without_metadatas = (
                    [embeddings[j] for j in empty_ids] if embeddings else None
                )
                ids_without_metadatas = [ids[j] for j in empty_ids]
                await asyncio.to_thread(
                    self._collection.upsert,
                    embeddings=embeddings_without_metadatas,
                    documents=texts_without_metadatas,
                    ids=ids_without_metadatas,
                )
        else:
            await asyncio.to_thread(
                self._collection.upsert,
                embeddings=embeddings,
                documents=texts,
                ids=ids,
            )
        return ids

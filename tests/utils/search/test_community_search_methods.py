from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.nodes import CommunityNode
from graphiti_core.search.search_config import (
    CommunityReranker,
    CommunitySearchConfig,
    CommunitySearchMethod,
    SearchConfig,
)


@pytest.fixture
def community_search_client():
    shared, keyword, vector = [
        CommunityNode(uuid=name, name=name, group_id='group')
        for name in ('shared', 'keyword', 'vector')
    ]
    # Keep search orchestration, retrieval wrappers and reranking real; only
    # replace the external driver operations and embedding generation.
    operations = SimpleNamespace(
        community_fulltext_search=AsyncMock(return_value=[shared, keyword]),
        community_similarity_search=AsyncMock(return_value=[shared, vector]),
        get_embeddings_for_communities=AsyncMock(
            return_value={'shared': [1.0, 0.0], 'keyword': [0.0, 1.0]}
        ),
    )
    driver = Mock(spec=GraphDriver)
    driver.provider = GraphProvider.NEO4J
    driver.search_interface = operations
    embedder = Mock(spec=EmbedderClient)
    embedder.create = AsyncMock(return_value=[1.0, 0.0])
    client = Graphiti(
        graph_driver=driver,
        llm_client=Mock(spec=LLMClient),
        embedder=embedder,
        cross_encoder=Mock(spec=CrossEncoderClient),
    )
    return client, operations


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'methods, expected_uuids, expected_scores',
    [
        ([], [], []),
        ([CommunitySearchMethod.bm25], ['shared', 'keyword'], [1.0, 0.5]),
        ([CommunitySearchMethod.cosine_similarity], ['shared', 'vector'], [1.0, 0.5]),
        (
            [CommunitySearchMethod.bm25, CommunitySearchMethod.cosine_similarity],
            ['shared', 'keyword', 'vector'],
            [2.0, 0.5, 0.5],
        ),
        (
            [CommunitySearchMethod.bm25, CommunitySearchMethod.bm25],
            ['shared', 'keyword'],
            [1.0, 0.5],
        ),
    ],
)
async def test_community_results_only_use_configured_methods(
    community_search_client, methods, expected_uuids, expected_scores
):
    client, operations = community_search_client
    config = SearchConfig(
        community_config=CommunitySearchConfig(search_methods=methods, sim_min_score=0.8),
        limit=3,
    )

    results = await client.search_('find communities', config=config, group_ids=['group'])

    assert [node.uuid for node in results.communities] == expected_uuids
    assert results.community_reranker_scores == expected_scores
    if CommunitySearchMethod.bm25 in methods:
        operations.community_fulltext_search.assert_awaited_once_with(
            client.driver, 'find communities', ['group'], 6
        )
    else:
        operations.community_fulltext_search.assert_not_called()
    if CommunitySearchMethod.cosine_similarity in methods:
        operations.community_similarity_search.assert_awaited_once_with(
            client.driver, [1.0, 0.0], ['group'], 6, 0.8
        )
        client.embedder.create.assert_awaited_once_with(input_data=['find communities'])
    else:
        operations.community_similarity_search.assert_not_called()
        client.embedder.create.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'method, unavailable_operation, expected_uuids',
    [
        (
            CommunitySearchMethod.bm25,
            'community_similarity_search',
            ['shared', 'keyword'],
        ),
        (
            CommunitySearchMethod.cosine_similarity,
            'community_fulltext_search',
            ['shared', 'vector'],
        ),
    ],
)
async def test_community_search_does_not_require_excluded_backend(
    community_search_client, method, unavailable_operation, expected_uuids
):
    client, operations = community_search_client
    getattr(operations, unavailable_operation).side_effect = RuntimeError('backend unavailable')

    results = await client.search_(
        'find communities',
        config=SearchConfig(community_config=CommunitySearchConfig(search_methods=[method])),
        group_ids=['group'],
    )

    assert [node.uuid for node in results.communities] == expected_uuids
    assert results.community_reranker_scores == [1.0, 0.5]


@pytest.mark.asyncio
@pytest.mark.parametrize('limit, min_score', [(1, 0.0), (3, 0.75)])
async def test_hybrid_community_search_preserves_limit_and_score_filter(
    community_search_client, limit, min_score
):
    client, _ = community_search_client
    results = await client.search_(
        'find communities',
        config=SearchConfig(
            community_config=CommunitySearchConfig(
                search_methods=[
                    CommunitySearchMethod.bm25,
                    CommunitySearchMethod.cosine_similarity,
                ]
            ),
            limit=limit,
            reranker_min_score=min_score,
        ),
        group_ids=['group'],
    )

    assert [node.uuid for node in results.communities] == ['shared']
    assert results.community_reranker_scores == [2.0]


@pytest.mark.asyncio
async def test_bm25_community_mmr_still_embeds_query(community_search_client):
    client, operations = community_search_client
    operations.community_similarity_search.side_effect = RuntimeError('backend unavailable')
    results = await client.search_(
        'find communities',
        config=SearchConfig(
            community_config=CommunitySearchConfig(
                search_methods=[CommunitySearchMethod.bm25],
                reranker=CommunityReranker.mmr,
                mmr_lambda=1.0,
            ),
        ),
        group_ids=['group'],
    )

    assert [node.uuid for node in results.communities] == ['shared', 'keyword']
    assert results.community_reranker_scores == [1.0, 0.0]
    client.embedder.create.assert_awaited_once_with(input_data=['find communities'])
    operations.community_similarity_search.assert_not_called()

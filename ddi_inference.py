# =============================================================================
# DDI Inference: 학습된 모델로 약물 상호작용 예측
# Chapter 4 코드에 이어서 실행하세요 (state, dataset_splits, drugs_lookup 필요)
# =============================================================================

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import linen as nn


# =============================================================================
# 1. Inference 함수 정의
# =============================================================================

def predict_interaction(state, graph, node_id_a, node_id_b):
    """
    두 약물 노드 간의 상호작용 확률을 반환합니다.

    Args:
        state      : 학습된 TrainState
        graph      : jraph.GraphsTuple (train graph)
        node_id_a  : 약물 A의 node index (int)
        node_id_b  : 약물 B의 node index (int)

    Returns:
        float: 상호작용 확률 (0~1)
    """
    pair = jnp.array([[node_id_a, node_id_b]])
    score = state.apply_fn(
        {"params": state.params},
        graph,
        pair,
        is_training=False,
        is_pred=True,
    )
    prob = float(nn.sigmoid(score))
    return prob


def predict_top_k_interactions(state, graph, node_ids, drugs_lookup, k=20, batch_size=512):
    """
    주어진 노드 집합에서 상호작용 확률이 높은 상위 K개 약물 쌍을 반환합니다.

    Args:
        state       : 학습된 TrainState
        graph       : jraph.GraphsTuple
        node_ids    : 예측할 노드 id 리스트 (없으면 전체)
        drugs_lookup: node_id → drug_name 매핑 DataFrame
        k           : 상위 몇 개 반환할지
        batch_size  : 한 번에 처리할 pair 수

    Returns:
        pd.DataFrame: 상위 K개 약물 쌍과 예측 확률
    """
    # 모든 가능한 약물 쌍 생성 (upper triangle, self-loop 제외)
    pairs = []
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            pairs.append([node_ids[i], node_ids[j]])
    pairs = np.array(pairs)

    print(f"총 {len(pairs):,}개 약물 쌍에 대해 예측 중...")

    # 배치 단위로 inference
    all_probs = []
    for start in range(0, len(pairs), batch_size):
        batch = jnp.array(pairs[start:start + batch_size])
        scores = state.apply_fn(
            {"params": state.params},
            graph,
            batch,
            is_training=False,
            is_pred=True,
        )
        probs = nn.sigmoid(scores)
        all_probs.append(np.array(probs))

    all_probs = np.concatenate(all_probs)

    # 결과 DataFrame 구성
    name_lookup = drugs_lookup.set_index("node_id")["drug_name"].to_dict()
    results = pd.DataFrame({
        "node_id_a": pairs[:, 0],
        "node_id_b": pairs[:, 1],
        "drug_a": [name_lookup.get(i, f"node_{i}") for i in pairs[:, 0]],
        "drug_b": [name_lookup.get(j, f"node_{j}") for j in pairs[:, 1]],
        "interaction_prob": all_probs,
    })

    results = results.sort_values("interaction_prob", ascending=False).reset_index(drop=True)
    return results.head(k)


def predict_for_drug(state, graph, drug_name, drugs_lookup, top_k=10):
    """
    특정 약물과 상호작용 가능성이 높은 다른 약물들을 반환합니다.

    Args:
        state      : 학습된 TrainState
        graph      : jraph.GraphsTuple
        drug_name  : 약물 이름 (문자열, 부분 매칭 가능)
        drugs_lookup: node_id → drug_name 매핑 DataFrame
        top_k      : 상위 몇 개 반환할지

    Returns:
        pd.DataFrame: 해당 약물과 상호작용 가능성 높은 약물 목록
    """
    # 약물 이름으로 node_id 찾기 (부분 매칭)
    match = drugs_lookup[drugs_lookup["drug_name"].str.contains(drug_name, case=False, na=False)]
    if match.empty:
        print(f"'{drug_name}' 약물을 찾을 수 없습니다.")
        print("사용 가능한 약물 목록 일부:")
        print(drugs_lookup["drug_name"].sample(10).tolist())
        return None

    query_node = int(match.iloc[0]["node_id"])
    query_drug_name = match.iloc[0]["drug_name"]
    print(f"'{query_drug_name}' (node_id={query_node}) 기준으로 예측 중...")

    # 해당 약물과 나머지 모든 약물 쌍 생성
    all_node_ids = drugs_lookup["node_id"].values
    other_nodes = all_node_ids[all_node_ids != query_node]

    pairs = np.array([[query_node, n] for n in other_nodes])

    # Batch inference
    all_probs = []
    batch_size = 512
    for start in range(0, len(pairs), batch_size):
        batch = jnp.array(pairs[start:start + batch_size])
        scores = state.apply_fn(
            {"params": state.params},
            graph,
            batch,
            is_training=False,
            is_pred=True,
        )
        probs = nn.sigmoid(scores)
        all_probs.append(np.array(probs))

    all_probs = np.concatenate(all_probs)

    name_lookup = drugs_lookup.set_index("node_id")["drug_name"].to_dict()
    results = pd.DataFrame({
        "query_drug": query_drug_name,
        "target_node_id": other_nodes,
        "target_drug": [name_lookup.get(n, f"node_{n}") for n in other_nodes],
        "interaction_prob": all_probs,
    })

    results = results.sort_values("interaction_prob", ascending=False).reset_index(drop=True)
    return results.head(top_k)


def check_known_vs_predicted(state, graph, dataset_splits, drugs_lookup, n_samples=10):
    """
    실제 알려진 상호작용(positive edges)과 모델 예측값을 비교합니다.
    True Positive(실제로 있는 상호작용)와 False Negative(놓친 상호작용)를 확인.

    Args:
        state          : 학습된 TrainState
        graph          : jraph.GraphsTuple
        dataset_splits : train/valid/test splits
        drugs_lookup   : node_id → drug_name 매핑 DataFrame
        n_samples      : 확인할 샘플 수

    Returns:
        pd.DataFrame: 예측 결과와 실제 label 비교
    """
    test_pos = dataset_splits["test"].pairs.pos[:n_samples]
    test_neg = dataset_splits["test"].pairs.neg[:n_samples]

    all_pairs = np.concatenate([test_pos, test_neg], axis=0)
    labels = np.array([1] * len(test_pos) + [0] * len(test_neg))

    scores = state.apply_fn(
        {"params": state.params},
        graph,
        jnp.array(all_pairs),
        is_training=False,
        is_pred=True,
    )
    probs = np.array(nn.sigmoid(scores))

    name_lookup = drugs_lookup.set_index("node_id")["drug_name"].to_dict()
    results = pd.DataFrame({
        "drug_a": [name_lookup.get(p[0], f"node_{p[0]}") for p in all_pairs],
        "drug_b": [name_lookup.get(p[1], f"node_{p[1]}") for p in all_pairs],
        "true_label": labels,
        "predicted_prob": probs,
        "predicted_label": (probs >= 0.5).astype(int),
        "correct": ((probs >= 0.5).astype(int) == labels),
    })

    return results


# =============================================================================
# 2. 실행 예시
# =============================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 아래 코드는 chapter_4_graphs.py 를 실행한 뒤, state / dataset_splits /
    # drugs_lookup 변수가 메모리에 있는 상태에서 실행하세요.
    # Colab에서는 같은 노트북 셀 안에 붙여넣기 하면 됩니다.
    # ------------------------------------------------------------------

    graph = dataset_splits["train"].graph

    # ── 예시 1: 특정 두 약물 간 상호작용 확률 ──────────────────────────
    print("=" * 60)
    print("예시 1: 두 약물 간 상호작용 확률")
    print("=" * 60)

    node_a = int(dataset_splits["train"].pairs.pos[0, 0])
    node_b = int(dataset_splits["train"].pairs.pos[0, 1])
    name_lookup = drugs_lookup.set_index("node_id")["drug_name"].to_dict()

    prob = predict_interaction(state, graph, node_a, node_b)
    print(f"  {name_lookup.get(node_a, f'node_{node_a}')} ↔ "
          f"{name_lookup.get(node_b, f'node_{node_b}')}")
    print(f"  예측 상호작용 확률: {prob:.4f} ({'상호작용 있음' if prob >= 0.5 else '상호작용 없음'})")

    # ── 예시 2: 특정 약물과 상호작용 가능성 높은 약물 TOP 10 ────────────
    print("\n" + "=" * 60)
    print("예시 2: 특정 약물 기준 상호작용 가능성 TOP 10")
    print("=" * 60)

    result_df = predict_for_drug(
        state, graph,
        drug_name="Warfarin",   # 약물 이름 변경 가능
        drugs_lookup=drugs_lookup,
        top_k=10,
    )
    if result_df is not None:
        print(result_df.to_string(index=False))

    # ── 예시 3: 전체 약물 중 상호작용 확률 TOP 20 쌍 ────────────────────
    print("\n" + "=" * 60)
    print("예시 3: 상호작용 확률 상위 20개 약물 쌍")
    print("=" * 60)

    sample_node_ids = drugs_lookup["node_id"].sample(50, random_state=42).tolist()
    top_pairs = predict_top_k_interactions(
        state, graph,
        node_ids=sample_node_ids,
        drugs_lookup=drugs_lookup,
        k=20,
    )
    print(top_pairs[["drug_a", "drug_b", "interaction_prob"]].to_string(index=False))

    # ── 예시 4: 실제 label vs 모델 예측 비교 ────────────────────────────
    print("\n" + "=" * 60)
    print("예시 4: 실제 상호작용 vs 모델 예측 비교 (test set)")
    print("=" * 60)

    comparison = check_known_vs_predicted(
        state, graph,
        dataset_splits=dataset_splits,
        drugs_lookup=drugs_lookup,
        n_samples=10,
    )
    print(comparison.to_string(index=False))

    accuracy = comparison["correct"].mean()
    print(f"\n  샘플 정확도: {accuracy:.2%}")

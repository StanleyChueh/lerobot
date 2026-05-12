import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

@torch.no_grad()
def extract_semantic_embeddings_from_policy(policy, top_k_tokens=10, device=None):
    """
    從已經載入的 policy 中萃取 FFN value vectors 並計算對應的 Top Tokens 與 Logits。
    這支函數現在可以直接吃 policy 物件，方便我們在「修改權重前後」重複呼叫。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vlm_model = policy.model.vlm_with_expert.vlm
    text_model = policy.model.vlm_with_expert.get_vlm_model().text_model
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer

    W_out = vlm_model.lm_head.weight.detach().to(device=device, dtype=torch.float32)

    semantic_embeddings = []
    metadata = []
    global_id = 0
    num_layers = len(text_model.layers)

    for layer_idx in range(num_layers):
        # 取得該層的 down_proj 權重
        W_value = text_model.layers[layer_idx].mlp.down_proj.weight.detach().to(
            device=device, dtype=torch.float32
        ) 

        # 計算 Logits: [vocab_size, d_ff]
        token_logits = W_out @ W_value

        # 取 Top-k tokens
        top_logits, top_token_ids = torch.topk(token_logits, k=top_k_tokens, dim=0)

        top_logits_t = top_logits.transpose(0, 1).contiguous()
        top_token_ids_t = top_token_ids.transpose(0, 1).contiguous()
        weights = torch.softmax(top_logits_t, dim=1)  
        token_embs = W_out[top_token_ids_t]

        # Semantic embedding
        e_sem = (weights.unsqueeze(-1) * token_embs).sum(dim=1)
        e_sem = torch.nn.functional.normalize(e_sem, p=2, dim=1)

        e_sem_np = e_sem.cpu().numpy()
        top_token_ids_np = top_token_ids_t.cpu().numpy()
        top_logits_np = top_logits_t.cpu().numpy()

        for neuron_idx in range(W_value.shape[1]):
            token_ids = top_token_ids_np[neuron_idx].tolist()
            decoded_tokens = [
                tokenizer.decode([tok_id]).replace("\n", " ").strip()
                for tok_id in token_ids
            ]

            # 計算該神經元的 L2 Norm
            neuron_norm = W_value[:, neuron_idx].norm().item()

            semantic_embeddings.append(e_sem_np[neuron_idx])
            metadata.append(
                {
                    "global_id": global_id,
                    "layer": layer_idx,
                    "neuron": neuron_idx,
                    "top_token_ids": token_ids,
                    "top_tokens": decoded_tokens,
                    "top_logits": top_logits_np[neuron_idx].tolist(),
                    "l2_norm": neuron_norm
                }
            )
            global_id += 1

    semantic_embeddings = np.asarray(semantic_embeddings, dtype=np.float32)
    return semantic_embeddings, metadata, tokenizer


def run_activation_steering_demo():
    policy_path = "ethanCSL/svla_koch_pick_n_place_vla_steering_height_test2_unfrozen"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[*] Loading policy: {policy_path}")
    policy = SmolVLAPolicy.from_pretrained(policy_path)
    policy.eval()
    policy.to(device)

    # 1. 定義要干預的神經元 (Layer ID: [Neuron IDs])
    # 這裡選擇你之前感興趣的神經元
    steering_config = {
        1: [1222],
        3: [2003],
        5: [1877, 1904],
        15: [1157]
    }
    
    # 2. 設定干預數值 (Alpha)
    # 例如：將這些神經元的激活值強制固定為 10.0
    alpha_value = 10.0
    
    print(f"\n[STEP 1] Applying Activation Steering (Alpha={alpha_value})...")
    policy.set_activation_steering(
        steering_neurons=steering_config,
        alpha=alpha_value,
        record_debug=True  # 必須開啟才能記錄前後數值
    )

    # 3. 執行一次 Dummy Forward Pass
    # 激活干預只有在模型「跑起來」的時候才會發生
    print("[STEP 2] Running dummy inference to trigger hooks...")
    
    # 建立模擬輸入 (根據 SmolVLA 的輸入格式，這裡簡化處理)
    # 注意：實際使用時請帶入真實的 observation
    batch_size = 1
    dummy_observation = {
        "observation.image": torch.randn(batch_size, 3, 224, 224).to(device),
        "observation.state": torch.randn(batch_size, 6).to(device), # 假設 state dim 為 6
    }

    with torch.no_grad():
        # 執行一次動作預測，這會觸發 Hook
        _ = policy.select_action(dummy_observation)

    # 4. 印出干預後的調試資訊
    # 這會調用你提供的 print_activation_steering_debug 函式
    print("\n[STEP 3] Printing Steered Neuron Statistics:")
    policy.print_activation_steering_debug()

    # 5. (選用) 如果想看這些被干預神經元「代表什麼」
    # 我們可以抓取之前程式碼中的 metadata 來輔助閱讀
    try:
        from lerobot_reord_top_token_steering import extract_semantic_embeddings_from_policy
        print("[STEP 4] Correlating with Semantic Meaning (Top Tokens)...")
        _, metadata, _ = extract_semantic_embeddings_from_policy(policy, top_k_tokens=3, device=device)
        
        lookup = {(m["layer"], m["neuron"]): m for m in metadata}
        
        print("\n" + "=" * 80)
        print(f"{'Layer':<6} | {'Neuron':<7} | {'Fixed Alpha':<12} | {'Top Tokens (Semantic Meaning)'}")
        print("-" * 80)
        for layer_idx, neurons in steering_config.items():
            for n_id in neurons:
                info = lookup.get((layer_idx, n_id))
                tokens = ", ".join(info["top_tokens"]) if info else "Unknown"
                print(f"L{layer_idx:<5} | {n_id:<7} | {alpha_value:<12.1f} | [{tokens}]")
        print("=" * 80)
        
    except ImportError:
        print("\n[Note] Could not find semantic extraction function to print tokens.")

    # 6. 清除 Hook (還原模型)
    policy.clear_activation_steering()
    print("\n[*] Demo completed. Hooks cleared.")

if __name__ == "__main__":
    run_activation_steering_demo()


def print_steered_neurons_info(metadata, config, title="Target Neurons Info"):
    """
    針對 multi_layer_steering_config 中指定的神經元，
    印出其 L2 Norm、Max Logit 以及 Top Tokens。
    """
    print("\n" + "=" * 115)
    print(title)
    print("=" * 115)
    print(f"{'Layer':<6} | {'Neuron':<7} | {'Config Target':<14} | {'Norm (L2)':<10} | {'Max Logit':<10} | {'Top Tokens'}")
    print("-" * 115)

    # 建立查找表，方便快速抓取特定層與神經元的資訊
    lookup = {(m["layer"], m["neuron"]): m for m in metadata}

    for layer_idx, neurons in sorted(config.items()):
        for neuron_idx, strength in sorted(neurons.items()):
            info = lookup.get((layer_idx, neuron_idx))
            if info:
                tokens_str = ", ".join(info["top_tokens"])
                max_logit = info["top_logits"][0]
                l2_norm = info["l2_norm"]
                print(f"L{layer_idx:<5} | {neuron_idx:<7} | {strength:<14.2f} | {l2_norm:<10.4f} | {max_logit:<10.4f} | [{tokens_str}]")
            else:
                print(f"L{layer_idx:<5} | {neuron_idx:<7} | {strength:<14.2f} | Not found in metadata")
    print("=" * 115 + "\n")


####################################################################################################

def activation_steering_config_for_print(steering_neurons, alpha):
    """
    Convert activation-steering format:

        {layer_idx: [neuron_ids]}

    into print format:

        {layer_idx: {neuron_idx: alpha}}

    This is only for display. The alpha here is activation target,
    not value-vector norm.
    """
    return {
        layer_idx: {neuron_idx: alpha for neuron_idx in neuron_ids}
        for layer_idx, neuron_ids in steering_neurons.items()
    }


def _make_steering_key(intervention_name, steering_neurons, alpha):
    """
    Stable key used to avoid printing/registering the same steering setup
    every time record_loop starts.
    """
    return (
        intervention_name,
        float(alpha),
        tuple(
            (layer_idx, tuple(neuron_ids))
            for layer_idx, neuron_ids in sorted(steering_neurons.items())
        ),
    )


def setup_activation_steering_with_terminal_log(
    policy,
    steering_neurons,
    alpha,
    intervention_name="activation_steering",
    top_k_tokens=5,
    device=None,
    print_top_tokens=True,
    record_debug=True,
):
    """
    1. Print selected neurons with value-vector top tokens.
    2. Register paper-alike activation steering.
    3. Store persistent state on policy so logging does not repeat
       every time record_loop is entered.

    Important:
        Top tokens come from value vectors.
        Activation before/after is recorded later during forward pass.
    """
    if device is None:
        try:
            device = next(policy.parameters()).device
        except StopIteration:
            device = "cuda" if torch.cuda.is_available() else "cpu"

    steering_key = _make_steering_key(
        intervention_name=intervention_name,
        steering_neurons=steering_neurons,
        alpha=alpha,
    )

    already_configured = (
        getattr(policy, "_stanley_steering_key", None) == steering_key
    )

    if already_configured:
        print(
            "[STEERING] Same activation steering already configured. "
            "Skip setup/top-token logging."
        )
        return

    if hasattr(policy, "clear_activation_steering"):
        policy.clear_activation_steering()

    if alpha == 0.0:
        print("[BASELINE] alpha=0.0: no activation steering hook registered.")
        policy._stanley_steering_key = steering_key
        policy._stanley_steering_neurons = steering_neurons
        policy._stanley_steering_alpha = float(alpha)
        policy._stanley_activation_debug_printed = True
        return

    if not hasattr(policy, "set_activation_steering"):
        raise AttributeError(
            "Policy does not have set_activation_steering(). "
            "Make sure modeling_smolvla.py contains the hook-based activation steering code."
        )

    if print_top_tokens:
        _, metadata, _ = extract_semantic_embeddings_from_policy(
            policy,
            top_k_tokens=top_k_tokens,
            device=device,
        )

        print_steered_neurons_info(
            metadata,
            activation_steering_config_for_print(steering_neurons, alpha),
            title=(
                f"[SELECTED BEFORE STEERING] "
                f"{intervention_name}: Value-Vector Top Tokens"
            ),
        )

    policy.set_activation_steering(
        steering_neurons=steering_neurons,
        alpha=alpha,
        record_debug=record_debug,
    )

    policy._stanley_steering_key = steering_key
    policy._stanley_steering_neurons = steering_neurons
    policy._stanley_steering_alpha = float(alpha)
    policy._stanley_activation_debug_printed = False


def print_activation_steering_debug_once(
    policy,
    disable_debug_after_print=True,
):
    """
    Print selected neurons' before/after activation once.

    This must be called AFTER the first real forward pass, e.g. after:

        action_values = predict_action(...)

    because activation only exists during forward.
    """
    if getattr(policy, "_stanley_activation_debug_printed", False):
        return False

    if not hasattr(policy, "print_activation_steering_debug"):
        print("[DEBUG] Policy does not have print_activation_steering_debug().")
        return False

    policy.print_activation_steering_debug()
    policy._stanley_activation_debug_printed = True

    if (
        disable_debug_after_print
        and hasattr(policy, "set_activation_steering")
        and getattr(policy, "_stanley_steering_alpha", 0.0) != 0.0
        and hasattr(policy, "_stanley_steering_neurons")
    ):
        # Re-register hooks without debug recording.
        # Steering continues, but before/after activation records stop accumulating.
        policy.set_activation_steering(
            steering_neurons=policy._stanley_steering_neurons,
            alpha=policy._stanley_steering_alpha,
            record_debug=False,
        )
        print(
            "[STEERING DEBUG] Printed activation debug once. "
            "Continuing steering with debug recording disabled."
        )

    return True
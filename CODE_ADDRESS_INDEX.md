# CODE ADDRESS INDEX

Comprehensive repository address map. Updated to current line-level state.

Total files indexed: **59**

## `.github/workflows/sync-from-upstream.yml`
- Type: text
- Total lines: 19
- Address anchors:
  - L1: `name: Auto Sync from Upstream`
  - L2: `on:`
  - L6: `jobs:`

## `.gitignore`
- Type: text
- Total lines: 169
- Address anchors: none detected

## `.gitmodules`
- Type: text
- Total lines: 9
- Address anchors: none detected

## `.vscode/launch.json`
- Type: text
- Total lines: 26
- Address anchors: none detected

## `.vscode/settings.json`
- Type: text
- Total lines: 3
- Address anchors: none detected

## `LICENSE`
- Type: text
- Total lines: 202
- Address anchors: none detected

## `README.md`
- Type: text
- Total lines: 112
- Address anchors:
  - L1: `# Visual Execution Model (VEM)`
  - L9: `## Purpose`
  - L17: `## Vision`
  - L20: `## Goal`
  - L28: `## Technical Architecture (Concept Map)`
  - L30: `### 1) Spatio-Temporal Representation (Vision Encoder)`
  - L35: `### 2) Geometric Inductive Biases`
  - L40: `### 3) Continuous-Time Latent Dynamics`
  - L45: `### 4) Hierarchical Predictive Reasoning`
  - L50: `### 5) World Rendering and Latent Scene Composition`
  - L54: `### 6) Latent Planning & Decision Support`
  - L58: `### 7) Multi-Modal and Robustness Extensions`
  - L62: `### 8) Training Stack`
  - L70: `## Repository Workflow (Single Framework)`
  - L72: `### Configurations`
  - L76: `### Training Entrypoint`
  - L79: `# or`
  - L86: `### Practical Notes`
  - L96: `### Final Execution Checklist (Do This)`
  - L104: `## Roadmap Direction`

## `arc_eval.ipynb`
- Type: text
- Total lines: 252
- Address anchors: none detected

## `assets/hrm.png`
- Type: binary/non-utf8
- Total lines: 0
- Address anchors: n/a

## `assets/npyjs.js`
- Type: text
- Total lines: 176
- Address anchors: none detected

## `check_integrations.py`
- Type: text
- Total lines: 56
- Address anchors:
  - L8: `def load_config(path: str):`
  - L13: `def main():`

## `config/arch/hrm_v1.yaml`
- Type: text
- Total lines: 21
- Address anchors:
  - L1: `name: hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1`
  - L2: `loss:`
  - L6: `halt_exploration_prob: 0.1`
  - L7: `halt_max_steps: 16`
  - L9: `H_cycles: 2`
  - L10: `L_cycles: 2`
  - L12: `H_layers: 4`
  - L13: `L_layers: 4`
  - L15: `hidden_size: 512`
  - L16: `num_heads: 8  # min(2, hidden_size // 64)`
  - L17: `expansion: 4`
  - L19: `puzzle_emb_ndim: ${.hidden_size}`
  - L21: `pos_encodings: rope`

## `config/cfg_pretrain.yaml`
- Type: text
- Total lines: 31
- Address anchors:
  - L3: `defaults:`
  - L7: `hydra:`
  - L11: `data_path: data/arc-aug-1000`
  - L14: `global_batch_size: 768`
  - L16: `epochs: 100000`
  - L17: `eval_interval: 10000`
  - L18: `checkpoint_every_eval: True`
  - L20: `lr: 1e-4`
  - L21: `lr_min_ratio: 1.0`
  - L22: `lr_warmup_steps: 2000`
  - L25: `beta1: 0.9`
  - L26: `beta2: 0.95`
  - L27: `weight_decay: 0.1`
  - L28: `puzzle_emb_weight_decay: 0.1`
  - L31: `puzzle_emb_lr: 1e-2`

## `config/vjepa_10b.yaml`
- Type: text
- Total lines: 85
- Address anchors:
  - L2: `encoder:`
  - L14: `predictor:`
  - L72: `training:`

## `config/vjepa_micro.yaml`
- Type: text
- Total lines: 34
- Address anchors:
  - L2: `encoder:`
  - L14: `predictor:`
  - L27: `training:`

## `dataset/build_arc_dataset.py`
- Type: text
- Total lines: 291
- Address anchors:
  - L19: `class DataProcessConfig(BaseModel):`
  - L37: `class ARCPuzzle:`
  - L43: `def arc_grid_to_np(grid: List[List[int]]):`
  - L54: `def np_grid_to_seq_translational_augment(inp: np.ndarray, out: np.ndarray, do_translation: bool):`
  - L81: `def puzzle_hash(puzzle: dict):`
  - L83: `def _grid_hash(grid: np.ndarray):`
  - L98: `def convert_single_arc_puzzle(results: dict, default_name: str, puzzle: dict, aug_count: int, dest_mapping: Dict[str, Tuple[str, str]]):`
  - L122: `def _map_grid(grid: np.ndarray):`
  - L148: `def load_puzzles_arcagi(results: dict, dataset_path: str, config: DataProcessConfig):`
  - L184: `def convert_dataset(config: DataProcessConfig):`
  - L286: `def main(config: DataProcessConfig):`

## `dataset/build_maze_dataset.py`
- Type: text
- Total lines: 142
- Address anchors:
  - L22: `class DataProcessConfig(BaseModel):`
  - L30: `def convert_subset(set_name: str, config: DataProcessConfig):`
  - L89: `def _seq_to_numpy(seq):`
  - L136: `def preprocess_data(config: DataProcessConfig):`

## `dataset/common.py`
- Type: text
- Total lines: 51
- Address anchors:
  - L12: `class PuzzleDatasetMetadata(pydantic.BaseModel):`
  - L27: `def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:`
  - L50: `def inverse_dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:`

## `dataset/generate_dummy_data.py`
- Type: text
- Total lines: 29
- Address anchors:
  - L5: `def generate_dummy_video(path, frames=32, res=(224, 224)):`

## `dataset/video_dataset.py`
- Type: text
- Total lines: 108
- Address anchors:
  - L8: `class AdvancedVideoDataset(IterableDataset):`
  - L13: `def __init__(self,`
  - L33: `def _get_video_stream(self, path):`
  - L42: `def _generate_3d_block_mask(self):`
  - L66: `def __iter__(self):`
  - L106: `def get_dataloader(video_paths, batch_size=1, **kwargs):`

## `docs/FRONTIER_GAP_ANALYSIS.md`
- Type: text
- Total lines: 35
- Address anchors:
  - L1: `# Frontier Capability Gap Analysis and Implementation Plan`
  - L3: `## Scope`
  - L9: `## Comparison Matrix`
  - L20: `## Implemented in this change`
  - L22: `### 1) MCTS action prior upgrade`
  - L28: `## Why this is prioritized first`
  - L31: `## Next technical steps (ordered)`

## `docs/HUMAN_VISION_EXECUTION_EVAL_SPEC.md`
- Type: text
- Total lines: 48
- Address anchors:
  - L1: `# Human-Vision + Execution Evaluation Spec (Initial)`
  - L5: `## Purpose Alignment`
  - L13: `## Track A: Perception Robustness (implemented baseline)`
  - L27: `## Track B: World-Model Dynamics (implemented baseline)`
  - L39: `## Track C: Execution/Cognition (next)`
  - L44: `## Promotion Rule (Phase-1/2)`

## `docs/RIGOROUS_DEVELOPMENT_PROTOCOL.md`
- Type: text
- Total lines: 33
- Address anchors:
  - L1: `# Rigorous Development Protocol (Phase-1)`
  - L5: `## Gate A — Sanity / Determinism`
  - L10: `## Gate B — World-model Metrics`
  - L25: `## Gate C — Change Promotion`
  - L31: `## Notes`

## `evaluate.py`
- Type: text
- Total lines: 68
- Address anchors:
  - L13: `class EvalConfig(pydantic.BaseModel):`
  - L19: `def launch():`

## `evaluate_perception.py`
- Type: text
- Total lines: 90
- Address anchors:
  - L26: `def apply_perturbation(video: torch.Tensor, mode: str) -> torch.Tensor:`
  - L39: `def latent_consistency(model: VJEPA, video: torch.Tensor, perturbed: torch.Tensor) -> float:`
  - L46: `def main() -> None:`

## `evaluate_world_model.py`
- Type: text
- Total lines: 166
- Address anchors:
  - L27: `def set_seed(seed: int) -> None:`
  - L34: `class EvalManifest:`
  - L46: `def get_commit_hash(default: str = "unknown") -> str:`
  - L62: `def latent_rollout(`
  - L76: `def evaluate_metrics(model: VJEPA, device: torch.device, rollout_steps: int, num_actions: int) -> Dict[str, float]:`
  - L116: `def main() -> None:`

## `models/adaptive_depth.py`
- Type: text
- Total lines: 195
- Address anchors:
  - L24: `class AdaptiveDepthController(nn.Module):`
  - L41: `def __init__(`
  - L54: `def should_continue(`
  - L116: `class AdaptiveDepthWrapper(nn.Module):`
  - L131: `def __init__(`
  - L146: `def forward(`

## `models/common.py`
- Type: text
- Total lines: 32
- Address anchors:
  - L7: `def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):`

## `models/hrm/hrm_act_v1.py`
- Type: text
- Total lines: 283
- Address anchors:
  - L16: `class HierarchicalReasoningModel_ACTV1InnerCarry:`
  - L22: `class HierarchicalReasoningModel_ACTV1Carry:`
  - L31: `class HierarchicalReasoningModel_ACTV1Config(BaseModel):`
  - L60: `class HierarchicalReasoningModel_ACTV1Block(nn.Module):`
  - L61: `def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:`
  - L77: `def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:`
  - L86: `class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):`
  - L87: `def __init__(self, layers: List[HierarchicalReasoningModel_ACTV1Block]):`
  - L92: `def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:`
  - L102: `class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):`
  - L103: `def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:`
  - L146: `def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):`
  - L168: `def empty_carry(self, batch_size: int):`
  - L174: `def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel_ACTV1InnerCarry):`
  - L180: `def forward(self, carry: HierarchicalReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:`
  - L216: `class HierarchicalReasoningModel_ACTV1(nn.Module):`
  - L219: `def __init__(self, config_dict: dict):`
  - L225: `def puzzle_emb(self):`
  - L228: `def initial_carry(self, batch: Dict[str, torch.Tensor]):`
  - L240: `def forward(self, carry: HierarchicalReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:`

## `models/hybrid_ssm.py`
- Type: text
- Total lines: 228
- Address anchors:
  - L26: `class SelectiveSSM(nn.Module):`
  - L41: `def __init__(`
  - L85: `def forward(self, x: torch.Tensor) -> torch.Tensor:`
  - L141: `class HybridSSMAttentionBlock(nn.Module):`
  - L161: `def __init__(`
  - L199: `def forward(`

## `models/information_bottleneck.py`
- Type: text
- Total lines: 227
- Address anchors:
  - L28: `class VariationalInformationBottleneck(nn.Module):`
  - L42: `def __init__(`
  - L71: `def _reparameterize(`
  - L90: `def forward(`
  - L135: `class InformationBottleneckAttention(nn.Module):`
  - L150: `def __init__(`
  - L179: `def forward(`

## `models/layers.py`
- Type: text
- Total lines: 167
- Address anchors:
  - L13: `def flash_attn_func(q, k, v, causal=False):`
  - L29: `def _find_multiple(a, b):`
  - L33: `def rotate_half(x: torch.Tensor):`
  - L40: `def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):`
  - L53: `class CastedLinear(nn.Module):`
  - L54: `def __init__(self,`
  - L68: `def forward(self, input: torch.Tensor) -> torch.Tensor:`
  - L72: `class CastedEmbedding(nn.Module):`
  - L73: `def __init__(self,`
  - L86: `def forward(self, input: torch.Tensor) -> torch.Tensor:`
  - L90: `class RotaryEmbedding(nn.Module):`
  - L91: `def __init__(self, dim, max_position_embeddings, base, device=None):`
  - L104: `def forward(self):`
  - L108: `class Attention(nn.Module):`
  - L109: `def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):`
  - L122: `def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:`
  - L148: `class SwiGLU(nn.Module):`
  - L149: `def __init__(self, hidden_size: int, expansion: float):`
  - L156: `def forward(self, x):`
  - L161: `def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:`

## `models/losses.py`
- Type: text
- Total lines: 101
- Address anchors:
  - L11: `def s(x, epsilon=1e-30):`
  - L19: `def log_stablemax(x, dim=-1):`
  - L24: `def stablemax_cross_entropy(logits, labels, ignore_index: int = -100):`
  - L34: `def softmax_cross_entropy(logits, labels, ignore_index: int = -100):`
  - L40: `class ACTLossHead(nn.Module):`
  - L41: `def __init__(self, model: nn.Module, loss_type: str):`
  - L46: `def initial_carry(self, *args, **kwargs):`
  - L49: `def forward(`

## `models/multimodal_grounding.py`
- Type: text
- Total lines: 258
- Address anchors:
  - L24: `class ModalityEncoder(nn.Module):`
  - L37: `def __init__(`
  - L60: `def forward(`
  - L88: `class CrossModalAttention(nn.Module):`
  - L101: `def __init__(`
  - L122: `def forward(`
  - L158: `class MultiModalGrounding(nn.Module):`
  - L180: `def __init__(`
  - L213: `def forward(`

## `models/muon_optimizer.py`
- Type: text
- Total lines: 191
- Address anchors:
  - L26: `class Muon(Optimizer):`
  - L44: `def __init__(`
  - L67: `def step(self, closure=None):`
  - L137: `def _newton_schulz_orthogonalize(G: torch.Tensor, steps: int = 5) -> torch.Tensor:`
  - L170: `def _distributed_allreduce_grads(`

## `models/proper_equivariance.py`
- Type: text
- Total lines: 335
- Address anchors:
  - L27: `class SO3Rotation(nn.Module):`
  - L37: `def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:`
  - L74: `def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:`
  - L93: `class WignerDMatrices(nn.Module):`
  - L107: `def wigner_d_small(beta: torch.Tensor, l: int) -> torch.Tensor:`
  - L170: `def rotation_to_euler(R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:`
  - L187: `class ProperSE3EquivariantLayer(nn.Module):`
  - L204: `def __init__(`
  - L243: `def _positional_encoding(self, positions: torch.Tensor) -> torch.Tensor:`
  - L259: `def forward(`

## `models/sparse_embedding.py`
- Type: text
- Total lines: 132
- Address anchors:
  - L11: `class CastedSparseEmbedding(nn.Module):`
  - L12: `def __init__(self, num_embeddings: int, embedding_dim: int, batch_size: int, init_std: float, cast_to: torch.dtype):`
  - L28: `def forward(self, inputs: torch.Tensor) -> torch.Tensor:`
  - L41: `class CastedSparseEmbeddingSignSGD_Distributed(Optimizer):`
  - L42: `def __init__(`
  - L63: `def step(self, closure=None):  # type: ignore`
  - L98: `def _sparse_emb_signsgd_dist(`

## `models/spectral_conv.py`
- Type: text
- Total lines: 193
- Address anchors:
  - L25: `class GraphLaplacian(nn.Module):`
  - L34: `def __init__(self, k_neighbors: int = 8):`
  - L38: `def forward(self, x: torch.Tensor) -> torch.Tensor:`
  - L74: `class SpectralGraphConv(nn.Module):`
  - L88: `def __init__(`
  - L120: `def _chebyshev_polynomials(`
  - L159: `def forward(self, x: torch.Tensor) -> torch.Tensor:`

## `models/topological.py`
- Type: text
- Total lines: 215
- Address anchors:
  - L25: `class DifferentiableBettiNumbers(nn.Module):`
  - L41: `def __init__(`
  - L68: `def _compute_distance_matrix(self, x: torch.Tensor) -> torch.Tensor:`
  - L74: `def _soft_threshold(`
  - L87: `def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:`
  - L156: `class TopologicalAwareness(nn.Module):`
  - L171: `def __init__(`
  - L191: `def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:`

## `models/ttt_layer.py`
- Type: text
- Total lines: 206
- Address anchors:
  - L25: `class TTTLinear(nn.Module):`
  - L45: `def __init__(`
  - L74: `def forward(`
  - L148: `class TTTLinearWithAttention(nn.Module):`
  - L163: `def __init__(`
  - L191: `def forward(`

## `models/uncertainty.py`
- Type: text
- Total lines: 206
- Address anchors:
  - L24: `class VariationalLinear(nn.Module):`
  - L41: `def __init__(`
  - L61: `def forward(self, x: torch.Tensor) -> torch.Tensor:`
  - L78: `def kl_divergence(self) -> torch.Tensor:`
  - L106: `class UncertaintyQuantification(nn.Module):`
  - L121: `def __init__(`
  - L151: `def forward(`

## `models/vjepa/flow_matching.py`
- Type: text
- Total lines: 281
- Address anchors:
  - L28: `class SinusoidalTimeEmbedding(nn.Module):`
  - L31: `def __init__(self, dim: int):`
  - L35: `def forward(self, t: torch.Tensor) -> torch.Tensor:`
  - L53: `class VelocityField(nn.Module):`
  - L61: `def __init__(self, dim: int, hidden_dim: int, condition_dim: int):`
  - L89: `def forward(`
  - L125: `class ConditionalFlowMatching(nn.Module):`
  - L150: `def __init__(`
  - L165: `def forward(`
  - L211: `def sample(`
  - L257: `def sample_rectified(`

## `models/vjepa/gaussian_splatting.py`
- Type: text
- Total lines: 203
- Address anchors:
  - L28: `class LatentGaussianSplatting(nn.Module):`
  - L40: `def __init__(self, dim: int, num_gaussians: int = 256):`
  - L75: `def _parse_gaussians(self, params: torch.Tensor) -> dict:`
  - L118: `def _quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:`
  - L146: `def forward(`

## `models/vjepa/layers.py`
- Type: text
- Total lines: 157
- Address anchors:
  - L8: `class LieGroupEquivariantLayer(nn.Module):`
  - L14: `def __init__(self, dim: int, rank: int = 8):`
  - L31: `def forward(self, x: torch.Tensor, group_element: torch.Tensor) -> torch.Tensor:`
  - L50: `class LatentRayMarcher(nn.Module):`
  - L56: `def __init__(self, dim: int, num_samples: int = 16):`
  - L71: `def forward(self, latents: torch.Tensor, ray_dirs: torch.Tensor) -> torch.Tensor:`
  - L113: `def apply_rotary_pos_emb_3d(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):`
  - L114: `def rotate_half(x):`
  - L128: `class RotaryEmbedding3D(nn.Module):`
  - L129: `def __init__(self, dim: int, max_t: int, max_h: int, max_w: int, base: float = 10000.0, device=None):`
  - L141: `def _get_freqs(self, length: int, dim: int, device):`
  - L147: `def _build_cache(self, device):`
  - L152: `def forward(self, t: int, h: int, w: int) -> Tuple[torch.Tensor, torch.Tensor]:`

## `models/vjepa/losses.py`
- Type: text
- Total lines: 31
- Address anchors:
  - L4: `def vicreg_loss(x, y, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):`
  - L22: `def covariance_loss(z):`

## `models/vjepa/memory.py`
- Type: text
- Total lines: 392
- Address anchors:
  - L28: `class ResonatorNetwork(nn.Module):`
  - L49: `def __init__(`
  - L66: `def set_cleanup_memory(self, memory: torch.Tensor) -> None:`
  - L70: `def cleanup(self, x: torch.Tensor) -> torch.Tensor:`
  - L96: `def resonator_step(`
  - L119: `def _unbind(self, composite: torch.Tensor, key: torch.Tensor) -> torch.Tensor:`
  - L125: `def forward(`
  - L178: `class HolographicMemory(nn.Module):`
  - L198: `def __init__(`
  - L219: `def _bind_hrr(self, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:`
  - L225: `def _unbind_hrr(self, composite: torch.Tensor, key: torch.Tensor) -> torch.Tensor:`
  - L231: `def _bind_fhrr(self, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:`
  - L243: `def _unbind_fhrr(self, composite: torch.Tensor, key: torch.Tensor) -> torch.Tensor:`
  - L249: `def bind(self, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:`
  - L255: `def unbind(self, composite: torch.Tensor, key: torch.Tensor) -> torch.Tensor:`
  - L261: `def superpose(self, vectors: torch.Tensor, dim: int = 1) -> torch.Tensor:`
  - L284: `def forward(`
  - L312: `def retrieve(self, memory: torch.Tensor, key: torch.Tensor) -> torch.Tensor:`
  - L327: `def retrieve_with_cleanup(`
  - L348: `def multi_retrieve(`
  - L385: `def set_cleanup_memory(self, memory: torch.Tensor) -> None:`

## `models/vjepa/physics_engine.py`
- Type: text
- Total lines: 107
- Address anchors:
  - L6: `class HRMPhysicsODE(nn.Module):`
  - L12: `def __init__(self, dim: int, action_dim: Optional[int] = 128):`
  - L30: `def forward(self, t: float, z: torch.Tensor) -> torch.Tensor:`
  - L61: `class ContinuousTimeHRM(nn.Module):`
  - L67: `def __init__(self, dim: int, action_dim: int = 128):`
  - L71: `def forward(self, z: torch.Tensor, delta_t: torch.Tensor | float = 1.0, action: Optional[torch.Tensor] = None):`

## `models/vjepa/planning.py`
- Type: text
- Total lines: 467
- Address anchors:
  - L28: `class MCTSNode:`
  - L49: `def __init__(`
  - L67: `def mean_value(self) -> float:`
  - L74: `def is_expanded(self) -> bool:`
  - L79: `def effective_visits(self) -> int:`
  - L83: `def puct_score(self, parent_visits: int, c_puct: float = 1.41) -> float:`
  - L120: `class MCTS:`
  - L139: `def __init__(`
  - L160: `def _imagine_future(`
  - L192: `def _select(self, node: MCTSNode) -> MCTSNode:`
  - L222: `def _expand(`
  - L289: `def _backpropagate(self, node: MCTSNode, value: float) -> None:`
  - L315: `def _get_action_probabilities(self, root: MCTSNode, num_actions: int) -> torch.Tensor:`
  - L349: `def plan(`
  - L402: `class LatentPlannerMCTS:`
  - L417: `def __init__(`
  - L433: `def plan(`
  - L451: `def plan_with_uncertainty(`

## `models/vjepa/predictor.py`
- Type: text
- Total lines: 260
- Address anchors:
  - L22: `class VJEPAPredictorInner(nn.Module):`
  - L28: `def __init__(self,`
  - L156: `def forward(self,`

## `models/vjepa/symplectic_integrator.py`
- Type: text
- Total lines: 137
- Address anchors:
  - L31: `class SymplecticEulerIntegrator(nn.Module):`
  - L48: `def __init__(self, dim: int, action_dim: Optional[int] = None):`
  - L67: `def set_action(self, action: Optional[torch.Tensor]) -> None:`
  - L71: `def hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:`
  - L87: `def forward(`
  - L126: `def compute_energy(self, z: torch.Tensor) -> torch.Tensor:`

## `models/vjepa/utils.py`
- Type: text
- Total lines: 44
- Address anchors:
  - L3: `def get_block_mask(t, h, w, mask_ratio=0.6):`
  - L21: `def apply_mask(x, mask):`

## `models/vjepa/vit.py`
- Type: text
- Total lines: 94
- Address anchors:
  - L9: `class PatchEmbed3D(nn.Module):`
  - L15: `def __init__(self, patch_size=(2, 16, 16), in_chans=3, embed_dim=768):`
  - L20: `def forward(self, x):`
  - L30: `class VisionTransformerBlock(nn.Module):`
  - L31: `def __init__(self, dim, num_heads, expansion, norm_eps=1e-5):`
  - L43: `def _forward_inner(self, x, cos_sin):`
  - L48: `def forward(self, x, cos_sin):`
  - L54: `class VisionEncoder(nn.Module):`
  - L55: `def __init__(self,`
  - L86: `def forward(self, x):`

## `models/vjepa/vjepa_model.py`
- Type: text
- Total lines: 141
- Address anchors:
  - L12: `class VJEPA(nn.Module):`
  - L23: `def __init__(self,`
  - L81: `def update_target_encoder(self):`
  - L86: `def forward(self, batch: Dict[str, torch.Tensor]):`
  - L139: `class VisualExecutionModel(VJEPA):`

## `pretrain.py`
- Type: text
- Total lines: 453
- Address anchors:
  - L26: `class LossConfig(pydantic.BaseModel):`
  - L32: `class ArchConfig(pydantic.BaseModel):`
  - L39: `class PretrainConfig(pydantic.BaseModel):`
  - L74: `class TrainState:`
  - L84: `def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):`
  - L108: `def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):`
  - L162: `def cosine_schedule_with_warmup_lr_lambda(`
  - L172: `def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):`
  - L190: `def save_train_state(config: PretrainConfig, train_state: TrainState):`
  - L199: `def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):`
  - L209: `def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):`
  - L266: `def evaluate(config: PretrainConfig, train_state: TrainState, eval_loader: torch.utils.data.DataLoader, eval_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):`
  - L333: `def save_code_and_config(config: PretrainConfig):`
  - L359: `def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:`
  - L381: `def launch(hydra_config: DictConfig):`

## `puzzle_dataset.py`
- Type: text
- Total lines: 199
- Address anchors:
  - L14: `def _sample_batch(rng: np.random.Generator, group_order: np.ndarray, puzzle_indices: np.ndarray, group_indices: np.ndarray, start_index: int, global_batch_size: int):`
  - L41: `class PuzzleDatasetConfig(pydantic.BaseModel):`
  - L53: `class PuzzleDataset(IterableDataset):`
  - L54: `def __init__(self, config: PuzzleDatasetConfig, split: str = "train"):`
  - L68: `def _load_metadata(self) -> PuzzleDatasetMetadata:`
  - L72: `def _lazy_load_dataset(self):`
  - L95: `def _collate_batch(self, batch):`
  - L118: `def _iter_test(self):`
  - L151: `def _iter_train(self):`
  - L189: `def __iter__(self):`

## `puzzle_visualizer.html`
- Type: text
- Total lines: 426
- Address anchors: none detected

## `requirements.txt`
- Type: text
- Total lines: 16
- Address anchors: none detected

## `utils/functions.py`
- Type: text
- Total lines: 19
- Address anchors:
  - L5: `def load_model_class(identifier: str, prefix: str = "models."):`
  - L15: `def get_model_source_path(identifier: str, prefix: str = "models."):`

## `vjepa_train.py`
- Type: text
- Total lines: 221
- Address anchors:
  - L19: `def build_optimizer(model, config):`
  - L93: `class CombinedOptimizer:`
  - L98: `def __init__(self, optimizers):`
  - L101: `def zero_grad(self, set_to_none=False):`
  - L105: `def step(self, closure=None):`
  - L110: `def param_groups(self):`
  - L117: `def train(config_path="config/vjepa_micro.yaml"):`


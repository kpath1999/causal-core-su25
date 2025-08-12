import networkx as nx
import numpy as np
from collections import defaultdict, deque
import wandb
import scipy.stats
import torch

from stable_baselines3.common.callbacks import BaseCallback

from causal_world.intervention_actors import GoalInterventionActorPolicy, RandomInterventionActorPolicy, \
    VisualInterventionActorPolicy, RigidPoseInterventionActorPolicy, PhysicalPropertiesInterventionActorPolicy

from causal_learning import (
    CausalGNN, CausalGraph, CausalGraphNode,
    CausalLearningDashboard, CausalInterpretability,
    CausalLearningMonitor
)

class InterventionNode:
    def __init__(self, actor_class, params=None, node_id=None):
        self.actor_class = actor_class
        self.params = params or {}
        self.node_id = node_id or f"{actor_class.__name__}_{id(self)}"

        # Enhanced statistical tracking
        self.activation_strength = 0.5
        self.reward_history = deque(maxlen=50)  # Increased history for better statistical power
        self.success_rate = 0.0
        self.causal_impact = 0.0
        self.effect_size = 0.0
        self.p_value = 1.0  # Track statistical significance
        self.confidence_interval = (0.0, 0.0)  # 95% confidence interval for causal impact

        # Graph connectivity
        self.prerequisites = []
        self.conflicts = []

        # Enhanced performance tracking
        self.times_activated = 0
        self.last_performance = 0.0
        self.activation_attempts = 0
        self.baseline_comparison_rewards = deque(maxlen=50)  # Increased for better comparison
        self.intervention_rewards = deque(maxlen=50)  # Increased for better comparison
        self.learning_progress = deque(maxlen=20)  # Track rate of improvement
        self.mastery_threshold = 0.8  # Threshold for considering a skill mastered

        # Causal learning components
        self.causal_mechanism = None
        self.state_embedding = None

        print(f"🔧 Created intervention node: {self.node_id} with params: {self.params}")
    
    def create_actor_instance(self):
        # create the actual intervention actor
        print(f"🎭 Creating actor instance for: {self.node_id}")
        return self.actor_class(**self.params)
    
    def update_performance(self, reward, success, was_active_this_episode):
        # update the node's performance metrics with proper causal attribution
        if was_active_this_episode:
            self.intervention_rewards.append(reward)
            self.reward_history.append(reward)
            self.times_activated += 1
            print(f"📊 Node {self.node_id}: ACTIVE episode - reward={reward:.6f}, "
                  f"total activations={self.times_activated}")
        else:
            self.baseline_comparison_rewards.append(reward)
            print(f"📊 Node {self.node_id}: INACTIVE episode - baseline reward={reward:.6f}")
        
        self.last_performance = reward
        
        # calculate success rate based on intervention episodes only
        if len(self.intervention_rewards) >= 2:
            self.success_rate = np.mean(self.intervention_rewards[-10:])
            print(f"📈 Node {self.node_id}: Success rate updated to {self.success_rate:.6f} "
                  f"(based on {len(self.intervention_rewards)} intervention episodes)")
    
    def calculate_causal_impact(self):
        """Enhanced causal impact calculation with statistical significance"""
        if len(self.intervention_rewards) >= 10 and len(self.baseline_comparison_rewards) >= 10:
            # Calculate means and standard deviations
            intervention_mean = np.mean(self.intervention_rewards)
            baseline_mean = np.mean(self.baseline_comparison_rewards)
            intervention_std = np.std(self.intervention_rewards)
            baseline_std = np.std(self.baseline_comparison_rewards)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt((intervention_std**2 + baseline_std**2) / 2)
            self.effect_size = (intervention_mean - baseline_mean) / pooled_std
            
            # Calculate t-statistic and p-value
            n1, n2 = len(self.intervention_rewards), len(self.baseline_comparison_rewards)
            t_stat = (intervention_mean - baseline_mean) / np.sqrt(intervention_std**2/n1 + baseline_std**2/n2)
            df = n1 + n2 - 2
            self.p_value = 2 * (1 - scipy.stats.t.cdf(abs(t_stat), df))
            
            # Calculate 95% confidence interval
            se = np.sqrt(intervention_std**2/n1 + baseline_std**2/n2)
            ci_lower = (intervention_mean - baseline_mean) - 1.96 * se
            ci_upper = (intervention_mean - baseline_mean) + 1.96 * se
            self.confidence_interval = (ci_lower, ci_upper)
            
            # Update causal impact
            self.causal_impact = intervention_mean - baseline_mean
            
            # Calculate learning progress
            if len(self.intervention_rewards) >= 2:
                recent_rewards = list(self.intervention_rewards)[-10:]
                self.learning_progress.append(np.mean(np.diff(recent_rewards)))
            
            print(f"🎯 Node {self.node_id}: CAUSAL IMPACT ANALYSIS")
            print(f"   Intervention mean: {intervention_mean:.6f} (n={n1})")
            print(f"   Baseline mean: {baseline_mean:.6f} (n={n2})")
            print(f"   Causal impact: {self.causal_impact:.6f}")
            print(f"   Effect size: {self.effect_size:.3f}")
            print(f"   p-value: {self.p_value:.6f}")
            print(f"   95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
            
            return self.causal_impact, self.effect_size, self.p_value
        else:
            print(f"🎯 Node {self.node_id}: Insufficient data for causal impact")
            return 0.0, 0.0, 1.0
    
    def is_mastered(self):
        """Determine if the intervention has been mastered based on multiple criteria"""
        if len(self.intervention_rewards) < 10:
            return False
            
        # Check statistical significance
        if self.p_value > 0.05:
            return False
            
        # Check effect size
        if self.effect_size < 0.5:  # Medium effect size threshold
            return False
            
        # Check learning progress
        if len(self.learning_progress) > 0:
            recent_progress = np.mean(list(self.learning_progress)[-5:])
            if recent_progress <= 0:  # No recent improvement
                return False
        
        # Check success rate
        success_rate = np.mean([r > 0.001 for r in self.intervention_rewards])
        if success_rate < self.mastery_threshold:
            return False
            
        return True

class GraphBasedCurriculumManager:
    def __init__(self, total_timesteps=50000):
        self.graph = nx.DiGraph()
        self.total_timesteps = total_timesteps
        self.baseline_reward = 0.0
        self.episode_count = 0
        
        # Enhanced curriculum phases
        self.baseline_episodes = 100      # Increased baseline period
        self.intervention_test_length = 50  # Longer intervention testing
        self.current_intervention_idx = 0
        self.episodes_in_current_test = 0
        
        # Adaptive parameters
        self.exploration_rate = 0.3
        self.learning_rate = 0.1
        self.reward_window = deque(maxlen=50)  # Increased window
        
        # Enhanced tracking
        self.intervention_test_order = []
        self.completed_tests = {}
        self.mastered_interventions = set()
        self.failed_interventions = set()
        self.learning_plateau_threshold = 0.001  # Threshold for detecting plateaus
        self.plateau_episodes = 0
        
        # Causal learning components
        self.causal_gnn = CausalGNN(node_dim=64, edge_dim=32)
        self.causal_graph = CausalGraph()
        self.interpreter = CausalInterpretability()
        self.monitor = CausalLearningMonitor()
        self.dashboard = CausalLearningDashboard()
        
        print(f"🚀 Initializing GraphBasedCurriculumManager with {total_timesteps} timesteps")
        print(f"📋 Experimental design: {self.baseline_episodes} baseline episodes, "
              f"{self.intervention_test_length} episodes per intervention test")
        
        self._build_intervention_graph()

    def _build_intervention_graph(self):
        # build on the intervention dependency graph
        print("🏗️ Building intervention dependency graph...")
        
        # Create intervention nodes
        nodes = {
            'goal_basic': InterventionNode(
                GoalInterventionActorPolicy,
                {},
                'goal_basic'
            ),
            'pose_position': InterventionNode(
                RigidPoseInterventionActorPolicy,
                {'positions': True, 'orientations': False},
                'pose_position'
            ),
            'pose_orientation': InterventionNode(
                RigidPoseInterventionActorPolicy,
                {'positions': False, 'orientations': True},
                'pose_orientation'
            ),
            'physics_friction': InterventionNode(
                PhysicalPropertiesInterventionActorPolicy,
                {'group': 'friction'},
                'physics_friction'
            ),
            'physics_mass': InterventionNode(
                PhysicalPropertiesInterventionActorPolicy,
                {'group': 'mass'},
                'physics_mass'
            ),
            'visual': InterventionNode(
                VisualInterventionActorPolicy,
                {},
                'visual'
            ),
            'random_full': InterventionNode(
                RandomInterventionActorPolicy,
                {},
                'random_full'
            )
        }

        # Add nodes to the graph
        for node_id, node in nodes.items():
            self.graph.add_node(node_id, node_obj=node)
            print(f"➕ Added node to graph: {node_id}")

        # Minimal dependencies for cleaner causal measurement
        dependencies = [
            ('goal_basic', 'pose_position'),  # Basic goal mastery before pose variations
        ]

        print("🔗 Setting up dependencies:")
        for prereq, dependent in dependencies:
            if prereq in nodes and dependent in nodes:
                self.graph.add_edge(prereq, dependent, edge_type='prerequisite')
                nodes[dependent].prerequisites.append(prereq)
                print(f"   {prereq} → {dependent}")

        # Conflicts to prevent simultaneous testing
        conflicts = [
            ('physics_mass', 'physics_friction'),  # Don't test both physics interventions simultaneously
            ('pose_position', 'pose_orientation'),        # Don't test both pose interventions simultaneously
        ]

        print("⚔️ Setting up conflicts:")
        for node1, node2 in conflicts:
            if node1 in nodes and node2 in nodes:
                self.graph.add_edge(node1, node2, edge_type='conflict')
                self.graph.add_edge(node2, node1, edge_type='conflict')
                nodes[node1].conflicts.append(node2)
                nodes[node2].conflicts.append(node1)
                print(f"   {node1} ⚔️ {node2}")

        # Initialize intervention test order
        self.intervention_test_order = list(nodes.keys())
        print(f"🧪 Intervention test order: {self.intervention_test_order}")
        
        print(f"✅ Graph construction complete: {len(self.graph.nodes())} nodes, {len(self.graph.edges())} edges")
    
    def get_eligible_interventions(self):
        """Enhanced eligibility checking with mastery verification"""
        print("\n🔍 Checking intervention eligibility...")
        eligible = []

        for node_id in self.graph.nodes():
            node = self.graph.nodes[node_id]['node_obj']
            
            # Skip if already mastered or failed
            if node_id in self.mastered_interventions:
                print(f"   ⏭️ {node_id} already mastered")
                continue
            if node_id in self.failed_interventions:
                print(f"   ⏭️ {node_id} previously failed")
                continue
            
            # Check prerequisites with enhanced criteria
            prerequisites_met = True
            prereq_details = []
            
            for prereq_id in node.prerequisites:
                if prereq_id in self.graph.nodes:
                    prereq_node = self.graph.nodes[prereq_id]['node_obj']
                    
                    # Enhanced prerequisite checking
                    has_sufficient_data = len(prereq_node.intervention_rewards) >= 10
                    shows_positive_impact = prereq_node.causal_impact > 0.001
                    is_statistically_significant = prereq_node.p_value < 0.05
                    has_adequate_effect = prereq_node.effect_size > 0.5
                    
                    prereq_satisfied = (has_sufficient_data and 
                                      shows_positive_impact and 
                                      is_statistically_significant and
                                      has_adequate_effect)
                    
                    prereq_details.append(
                        f"{prereq_id}(satisfied={prereq_satisfied}, "
                        f"data={len(prereq_node.intervention_rewards)}, "
                        f"impact={prereq_node.causal_impact:.6f}, "
                        f"p={prereq_node.p_value:.6f}, "
                        f"effect={prereq_node.effect_size:.3f})"
                    )
                    
                    if not prereq_satisfied:
                        prerequisites_met = False
                else:
                    prerequisites_met = False
                    prereq_details.append(f"{prereq_id}(NOT_FOUND)")

            print(f"   Node {node_id}: prerequisites_met={prerequisites_met}")
            if prereq_details:
                print(f"      Prerequisites: {', '.join(prereq_details)}")
            
            if prerequisites_met:
                eligible.append(node_id)
                print(f"   ✅ {node_id} is ELIGIBLE")
            else:
                print(f"   ❌ {node_id} is NOT eligible")

        # Enhanced fallback strategy
        if not eligible:
            print("🆘 No eligible interventions - checking for potential retries")
            # Consider retrying failed interventions with new parameters
            for node_id in self.failed_interventions:
                node = self.graph.nodes[node_id]['node_obj']
                if len(node.intervention_rewards) < 20:  # Give more chances if not tested enough
                    eligible.append(node_id)
                    print(f"   🔄 Retrying {node_id} with more episodes")
            
            # If still no eligible interventions, use basic ones
            if not eligible:
                basic_interventions = ['goal_basic', 'visual', 'physics_mass']
                eligible = [node_id for node_id in basic_interventions if node_id in self.graph.nodes()]
                print(f"   🆘 FALLBACK: Using basic interventions: {eligible}")

        print(f"🎯 Final eligible interventions: {eligible}")
        return eligible

    def select_active_interventions(self):
        """Enhanced intervention selection with adaptive testing"""
        print(f"\n🧪 EXPERIMENTAL PHASE SELECTION (Episode {self.episode_count})")
        
        # Phase 1: Extended baseline period
        if self.episode_count < self.baseline_episodes:
            print(f"📊 BASELINE PHASE: Episode {self.episode_count}/{self.baseline_episodes}")
            return []
        
        # Phase 2: Enhanced intervention testing
        eligible = self.get_eligible_interventions()
        if not eligible:
            print("❌ No eligible interventions - extending baseline period")
            return []
        
        # Adaptive test length based on learning progress
        test_episode = self.episode_count - self.baseline_episodes
        current_test_cycle = test_episode // self.intervention_test_length
        intervention_idx = current_test_cycle % len(eligible)
        episodes_in_current_test = test_episode % self.intervention_test_length
        
        current_intervention = eligible[intervention_idx]
        node = self.graph.nodes[current_intervention]['node_obj']
        
        # Check for learning plateaus
        if len(node.learning_progress) > 0:
            recent_progress = np.mean(list(node.learning_progress)[-5:])
            if abs(recent_progress) < self.learning_plateau_threshold:
                self.plateau_episodes += 1
                if self.plateau_episodes > 10:  # Plateau detected
                    print(f"📉 Learning plateau detected for {current_intervention}")
                    self.failed_interventions.add(current_intervention)
                    return self.select_active_interventions()  # Try next intervention
            else:
                self.plateau_episodes = 0
        
        print(f"🔬 INTERVENTION TEST PHASE:")
        print(f"   Test cycle: {current_test_cycle}")
        print(f"   Testing intervention: {current_intervention}")
        print(f"   Episodes in current test: {episodes_in_current_test}/{self.intervention_test_length}")
        
        # Adaptive intervention strength
        strength = 0.7  # Base strength
        if node.is_mastered():
            strength *= 1.2  # Increase difficulty for mastered interventions
        elif current_intervention in self.failed_interventions:
            strength *= 0.8  # Reduce difficulty for retries
        
        selected_intervention = {
            'node_id': current_intervention,
            'actor': node.create_actor_instance(),
            'strength': strength
        }
        
        print(f"   🚀 SELECTED: {current_intervention} (strength: {strength:.2f})")
        return [selected_intervention]

    def update_graph_with_performance(self, episode_reward, episode_success, active_node_ids):
        """Enhanced performance tracking with mastery detection"""
        print(f"\n📈 PERFORMANCE UPDATE (Episode {self.episode_count})")
        print(f"   Reward: {episode_reward:.6f}, Success: {episode_success}")
        print(f"   Active interventions: {active_node_ids}")
        
        self.episode_count += 1
        self.reward_window.append(episode_reward)
        old_baseline = self.baseline_reward
        self.baseline_reward = np.mean(list(self.reward_window)) if self.reward_window else 0.0
        
        print(f"   Baseline reward: {old_baseline:.6f} → {self.baseline_reward:.6f}")

        # Update nodes with enhanced tracking
        for node_id in self.graph.nodes():
            node = self.graph.nodes[node_id]['node_obj']
            was_active = node_id in active_node_ids
            
            node.update_performance(episode_reward, episode_success, was_active)
            
            # Enhanced causal impact calculation
            if self.episode_count % 10 == 0:
                causal_impact, effect_size, p_value = node.calculate_causal_impact()
                
                # Check for mastery
                if node.is_mastered():
                    print(f"   🎓 MASTERY ACHIEVED for {node_id}")
                    self.mastered_interventions.add(node_id)
                
                # Log significant findings with enhanced metrics
                if p_value < 0.05 and effect_size > 0.5:
                    print(f"   🎯 SIGNIFICANT EFFECT detected for {node_id}:")
                    print(f"      Impact: {causal_impact:.6f}")
                    print(f"      Effect size: {effect_size:.3f}")
                    print(f"      p-value: {p_value:.6f}")
                    print(f"      CI: [{node.confidence_interval[0]:.6f}, {node.confidence_interval[1]:.6f}]")
                    
                    # Update causal graph
                    self.update_causal_graph(node_id, causal_impact, effect_size, p_value)

    def update_causal_graph(self, node_id, causal_impact, effect_size, p_value):
        """Update causal graph with new causal relationships"""
        # Update node in causal graph
        if node_id not in self.causal_graph.nodes:
            self.causal_graph.nodes[node_id] = CausalGraphNode(
                node_type='intervention',
                properties={
                    'causal_impact': causal_impact,
                    'effect_size': effect_size,
                    'p_value': p_value
                }
            )
        
        # Update edges based on causal relationships
        for other_node_id in self.graph.nodes():
            if other_node_id != node_id:
                other_node = self.graph.nodes[other_node_id]['node_obj']
                if other_node.causal_impact > 0:
                    # Add edge if there's a significant causal relationship
                    self.causal_graph.edges[(node_id, other_node_id)] = {
                        'weight': effect_size,
                        'type': 'causal'
                    }

    def get_current_curriculum_config(self):
        """Generate curriculum configuration with experimental controls"""
        print(f"\n🎓 CURRICULUM CONFIG GENERATION (Episode {self.episode_count})")
        
        # Baseline period: no interventions
        if self.episode_count < self.baseline_episodes:
            print("   📊 BASELINE PERIOD: No interventions active")
            return {
                'intervention_actors': [],
                'actives': [],
                'active_node_ids': []
            }
        
        # Intervention testing period
        active_interventions = self.select_active_interventions()
        
        if not active_interventions:
            print("   🆘 No interventions selected - using fallback")
            return {
                'intervention_actors': [GoalInterventionActorPolicy()],
                'actives': [(0, self.total_timesteps, 1, 0)],
                'active_node_ids': ['goal_basic']
            }

        actors = []
        actives = []

        print("   🔧 Building curriculum configuration:")
        for intervention in active_interventions:
            actors.append(intervention['actor'])
            # Fixed frequency for consistent testing
            frequency = 1  # Apply every episode during test period
            actives.append((0, self.total_timesteps, frequency, 0))
            print(f"      {intervention['node_id']}: frequency={frequency}, "
                  f"strength={intervention['strength']:.3f}")

        config = {
            'intervention_actors': actors,
            'actives': actives,
            'active_node_ids': [intervention['node_id'] for intervention in active_interventions]
        }
        
        print(f"   ✅ Config ready: {len(actors)} actors, {len(config['active_node_ids'])} active nodes")
        return config

class GraphBasedCurriculumCallback(BaseCallback):
    def __init__(self, curriculum_manager: GraphBasedCurriculumManager,
                 curriculum_wrapper_env, adaptation_interval_episodes: int = 5, verbose=0):
        super().__init__(verbose)
        self.curriculum_manager = curriculum_manager
        self.curriculum_wrapper_env = curriculum_wrapper_env
        self.adaptation_interval_episodes = adaptation_interval_episodes
        self.episodes_since_adaptation = 0
        self.current_active_node_ids = []
        
        print(f"🔄 Curriculum callback initialized: adaptation every {adaptation_interval_episodes} episodes")

    def _on_step(self) -> bool:
        """Handle episode completion and performance tracking"""
        if self.locals.get("dones") is not None and self.locals["dones"][0]:
            self.episodes_since_adaptation += 1

            # Get episode performance
            if len(self.model.ep_info_buffer) > 0:
                episode_reward = self.model.ep_info_buffer[-1]['r']
                episode_success = episode_reward > 0.001
                
                print(f"\n📊 EPISODE COMPLETED:")
                print(f"   Episode: {self.curriculum_manager.episode_count + 1}")
                print(f"   Reward: {episode_reward:.6f}")
                print(f"   Success: {episode_success}")
                print(f"   Active interventions: {self.current_active_node_ids}")

                # Update graph with performance data
                self.curriculum_manager.update_graph_with_performance(
                    episode_reward, episode_success, self.current_active_node_ids
                )

            # Adapt curriculum at specified intervals
            if self.episodes_since_adaptation >= self.adaptation_interval_episodes:
                print(f"\n🔄 CURRICULUM ADAPTATION TRIGGERED")
                self.episodes_since_adaptation = 0
                self._adapt_curriculum()

        return True

    def _adapt_curriculum(self):
        """Update curriculum with enhanced error handling and logging"""
        print("🔄 Adapting curriculum...")
        try:
            new_curriculum_config = self.curriculum_manager.get_current_curriculum_config()

            # Update wrapper components
            print("🔧 Updating curriculum wrapper...")
            self.curriculum_wrapper_env.intervention_actors = new_curriculum_config['intervention_actors']
            self.curriculum_wrapper_env.actives = new_curriculum_config['actives']

            # Update internal curriculum object
            print("🔧 Updating internal curriculum object...")
            self.curriculum_wrapper_env.interventions_curriculum.intervention_actors = new_curriculum_config['intervention_actors']
            self.curriculum_wrapper_env.interventions_curriculum.actives = new_curriculum_config['actives']

            # Reinitialize curriculum actors
            print("🔧 Reinitializing curriculum actors...")
            self.curriculum_wrapper_env.interventions_curriculum.initialize_actors(
                env=self.curriculum_wrapper_env.env
            )

            # Update tracking
            self.current_active_node_ids = new_curriculum_config.get('active_node_ids', [])

            # Enhanced W&B logging
            if wandb.run:
                log_data = {
                    "curriculum/num_active_interventions": len(self.current_active_node_ids),
                    "curriculum/episode_count": self.curriculum_manager.episode_count,
                    "curriculum/baseline_reward": self.curriculum_manager.baseline_reward,
                    "curriculum/phase": "baseline" if self.curriculum_manager.episode_count < self.curriculum_manager.baseline_episodes else "intervention_testing"
                }
                
                # Log individual intervention metrics
                for node_id in self.curriculum_manager.graph.nodes():
                    node = self.curriculum_manager.graph.nodes[node_id]['node_obj']
                    log_data[f"intervention_{node_id}/causal_impact"] = node.causal_impact
                    log_data[f"intervention_{node_id}/times_activated"] = node.times_activated
                    log_data[f"intervention_{node_id}/intervention_episodes"] = len(node.intervention_rewards)
                    log_data[f"intervention_{node_id}/baseline_episodes"] = len(node.baseline_comparison_rewards)

                wandb.log(log_data, step=self.num_timesteps)

            print(f"✅ Curriculum updated successfully!")
            print(f"   📊 Active interventions: {len(self.current_active_node_ids)}")
            print(f"   🎯 Active nodes: {self.current_active_node_ids}")
            print(f"   📈 Baseline reward: {self.curriculum_manager.baseline_reward:.6f}")
            print(f"   🧪 Episode count: {self.curriculum_manager.episode_count}")

        except Exception as e:
            print(f"❌ Error updating curriculum: {e}")
            import traceback
            traceback.print_exc()
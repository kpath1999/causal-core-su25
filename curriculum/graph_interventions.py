import networkx as nx
import numpy as np
from collections import defaultdict, deque
import wandb

from causal_world.intervention_actors import GoalInterventionActorPolicy, RandomInterventionActorPolicy, \
    VisualInterventionActorPolicy, RigidPoseInterventionActorPolicy, PhysicalPropertiesInterventionActorPolicy

class InterventionNode:
    def __init__(self, actor_class, params=None, node_id=None):
        self.actor_class = actor_class
        self.params = params or {}
        self.node_id = node_id or f"{actor_class.__name__}_{id(self)}"

        # dynamic properties
        self.activation_strength = 0.1  # frequency of intervention

        self.reward_history = deque(maxlen=100)
        self.success_rate = 0.0
        self.causal_impact = 0.0    # estimated causal impact

        # graph connectivity
        self.prerequisites = []  # nodes that should be active before this one
        self.conflicts = []      # nodes that should not be active with this one
    
    def create_actor_instance(self):
        # create the actual intervention actor
        return self.actor_class(**self.params)
    
    def update_performance(self, reward, success):
        # update the node's performance metrics
        self.reward_history.append(reward)
        if len(self.reward_history) > 10:
            recent_rewards = list(self.reward_history)[-10:]
            self.success_rate = np.mean([r > 0 for r in recent_rewards])
    
    def calculate_causal_impact(self, baseline_reward):
        # estimate causal impact compared to baseline
        if len(self.reward_history) > 5:
            avg_reward = np.mean(list(self.reward_history)[-20:])
            self.causal_impact = avg_reward - baseline_reward
        return self.causal_impact
    
class GraphBasedCurriculumManager:
    def __init__(self, total_timesteps=50000):
        self.graph = nx.DiGraph()
        self.total_timesteps = total_timesteps
        self.baseline_reward = 0.0
        self.episode_count = 0

        # adaptive parameters
        self.exploration_rate = 0.3  # prob. of trying new interventions
        self.learning_rate = 0.05
        self.reward_window = deque(maxlen=50)

        self._build_intervention_graph()

    def _build_intervention_graph(self):
        # build on the intervention dependency graph

        # create intervention nodes
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
            'pose_full': InterventionNode(
                RigidPoseInterventionActorPolicy,
                {'positions': True, 'orientations': True},
                'pose_full'
            ),
            'physics_friction': InterventionNode(
                PhysicalPropertiesInterventionActorPolicy,
                {'group': 'mass'},
                'physics_friction'
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

        # add nodes to the graph
        for node_id, node in nodes.items():
            self.graph.add_node(node_id, node_obj=node)
        
        # define dependency edges (prerequite relationships)
        dependencies = [
            ('goal_basic', 'pose_position'),      # master basic goals before pose estimation
            ('pose_position', 'pose_full'),       # position before orientation
            ('pose_position', 'physics_mass'),    # position before mass changes
            ('physics_mass', 'physics_friction'), # mass before friction
            ('pose_full', 'visual'),              # full pose before visual variations
            ('physics_friction', 'random_full')   # physics mastery before full randomization
        ]

        for prereq, dependent in dependencies:
            self.graph.add_edge(prereq, dependent, edge_type='prerequisite')
            nodes[dependent].prerequisites.append(prereq)
        
        # define conflict relationships (mutually exlusive relationships)
        conflicts = [
            ('physics_mass', 'physics_friction')    # mass vs friction
        ]

        for node1, node2 in conflicts:
            self.graph.add_edge(node1, node2, edge_type='conflict')
            nodes[node1].conflicts.append(node2)
            nodes[node2].conflicts.append(node1)
    
    def get_eligible_interventions(self):
        # interventions that can get activated based on graph constraints
        eligible = []

        for node_id in self.graph.nodes():
            node = self.graph.nodes[node_id]['node_obj']

            # check if all prerequisites are met
            prerequisites_met = True
            for prereq_id in node.prerequisites:
                prereq_node = self.graph.nodes[prereq_id]['node_obj']
                if prereq_node.success_rate < 0.6:
                    prerequisites_met = False
                    break
            
            if prerequisites_met:
                eligible.append(node)
        
        return eligible
    
    def select_active_interventions(self):
        # dynamically select which interventions should be active
        eligible = self.get_eligible_interventions()
        active_interventions = []

        # multi-armed bandit approach for intervention selection
        for node_id in eligible:
            node = self.graph.nodes[node_id]['node_obj']
        
            # upper confidence bound (UCB) selection
            if len(node.reward_history) < 5:
                # exploration: try interventions with less data
                selection_score = float('inf')
            else:
                avg_reward = np.mean(list(node.reward_history)[-20:])
                confidence = np.sqrt(2 * np.log(self.episode_count) / len(node.reward_history))
                selection_score = avg_reward + confidence
            
            # adaptive activation strength based on performance
            if node.causal_impact > 0:
                node.activation_strength = min(1.0, node.activation_strength + self.learning_rate)
            else:
                node.activation_strength = max(0.1, node.activation_strength - self.learning_rate)
            
            # decide if this intervention should be active
            if (selection_score > self.baseline_reward or np.random.random() < self.exploration_rate):
                # check conflicts
                has_conflict = False
                for conflict_id in node.conflicts:
                    if any(node['node_id'] == conflict_id for node in active_interventions):
                        has_conflict = True
                        break
                
                if not has_conflict:
                    active_interventions.append({
                        'node_id': node_id,
                        'actor': node.create_actor_instance(),
                        'activation_strength': node.activation_strength
                    })
        
        return active_interventions
    
    def update_graph_with_performance(self, episode_reward, episode_success, active_node_ids):
        # update graph nodes based on episode performance
        self.episode_count += 1
        self.reward_window.append(episode_reward)
        self.baseline_reward = np.mean(list(self.reward_window)) if self.reward_window else 0.0

        # update performance for active interventions
        for node_id in active_node_ids:
            node = self.graph.nodes[node_id]['node_obj']
            node.update_performance(episode_reward, episode_success)
            node.calculate_causal_impact(self.baseline_reward)
        
        # decay exploration rate over time
        self.exploration_rate = max(0.1, self.exploration_rate * 0.999)

    def get_current_curriculum_config(self):
        # generate curriculum config for CurriculumWrapper
        active_interventions = self.select_active_interventions()

        if not active_interventions:
            # fallback to a basic goal intervention
            return {
                'intervention_actors': [GoalInterventionActorPolicy()],
                'actives': [(0, self.total_timesteps, 1, 0)]
            }
        
        actors = []
        actives = []

        for intervention in active_interventions:
            actors.append(intervention['actor'])

            # converting activation strength to intervention frequency
            # strength = 1.0 means every episode, strength = 0.5 means every 2 episodes
            frequency = max(1, int(1.0 / intervention['strength']))
            actives.append((0, self.total_timesteps, frequency, 0))
        
        return {
            'intervention_actors': actors,
            'actives': actives,
            'active_interventions': [node['node_id'] for node in active_interventions]
        }

class GraphBasedCurriculumCallback(BaseCallback):
    def __init__(self, curriculum_manager: GraphBasedCurriculumManager,
                 curriculum_wrapper_env, adaptation_interval_episodes: int = 25, verbose=0):
        super().__init__(verbose)
        self.curriculum_manager = curriculum_manager
        self.curriculum_wrapper_env = curriculum_wrapper_env
        self.adaptation_interval_episodes = adaptation_interval_episodes
        self.episodes_since_adaptation = 0
        self.current_active_node_ids = []
    
    def _on_step(self) -> bool:
        # check if episode is finished
        if self.locals.get("dones") is not None and self.locals["dones"][0]:
            self.episodes_since_adaptation += 1

            # get episode performance
            if len(self.model.ep_info_buffer) > 0:
                episode_reward = self.model.ep_info_buffer[-1]['r']
                episode_success = episode_reward > 0  # TODO: adjust based on success criteria

                # update graph with performance data
                self.curriculum_manager.update_graph_with_performance(
                    episode_reward, episode_success, self.current_active_node_ids
                )
            
            # adapt curriculum periodically
            if self.episodes_since_adaptation >= self.adaptation_interval_episodes:
                self.episodes_since_adaptation = 0
                self._adapt_curriculum()
        
        return True

    def _adapt_curriculum(self):
        # update curriculum based on graph analysis
        new_curriculum_config = self.curriculum_manager.get_current_curriculum_config()

        # update wrapper
        self.curriculum_wrapper_env.intervention_actors = new_curriculum_config['intervention_actors']
        self.curriculum_wrapper_env.actives = new_curriculum_config['actives']
        self.current_active_node_ids = new_curriculum_config.get('active_node_ids', [])

        # reinitialize actors
        for actor in self.curriculum_wrapper_env.intervention_actors:
            actor.initialize(self.curriculum_wrapper_env.env)
        
        # log to wandb
        if wandb.run:
            # log active interventions and their strengths
            active_interventions = {}
            for node_id in self.current_active_node_ids:
                node = self.curriculum_manager.graph.nodes[node_id]['node_obj']
                active_interventions[f"intervention_{node_id}"] = node.activation_strength
                active_interventions[f"causal_impact_{node_id}"] = node.causal_impact
            
            wandb.log({
                "curriculum/num_active_interventions": len(self.current_active_node_ids),
                "curriculum/exploration_rate": self.curriculum_manager.exploration_rate,
                "curriculum/baseline_reward": self.curriculum_manager.baseline_reward,
                **active_interventions
            }, step=self.num_timesteps)
        
        print(f"Updated curriculum: {len(self.current_active_node_ids)} active interventions")
        print(f"Active nodes: {self.current_active_node_ids}")
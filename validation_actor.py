import numpy as np
from causal_world.intervention_actors.base_actor import BaseInterventionActorPolicy

class ValidationInterventionActorPolicy(BaseInterventionActorPolicy):

    def __init__(self, seed=0, intervention_probability=0.7, **kwargs):
        """
        here, the intervention actor creates a validation set by combining multiple
        intervention types in a deterministic, seed-based manner.

        :param seed: (int) the random seed for deterministic intervention sampling
        :param intervention_probability: (float) probability of applying each intervention type
        :param kwargs: additional parameters
        """
        super(ValidationInterventionActorPolicy, self).__init__()
        self.seed = seed
        self.intervention_probability = intervention_probability
        self.task_intervention_space = None
        self.goal_sampler_function = None
        self._rng = np.random.RandomState(seed)

        # track intervention call count for seed variation
        self._call_count = 0
    
    def initialize(self, env):
        """
        initialize the validation actor with environment-specific functions

        :param env: (causal_world.env.CausalWorld) the environment
        :return:
        """
        self.task_intervention_space = env.get_variable_space_used()
        self.goal_sampler_function = env.sample_new_goal
        return
    
    def _act(self, variables_dict):
        """
        create combined interventions from multiple intervention types

        :param variables_dict: current variables dictionary
        :return: combined interventions dictionary
        """
        # use the call count to vary seed while maintaining determinism
        current_seed = self.seed + self._call_count
        self._rng = np.random.RandomState(current_seed)
        self._call_count += 1

        interventions_dict = dict()

        # 1. goal intervention (30% probability)
        if self._rng.random() < 0.3 and self.goal_sampler_function:
            goal_interventions = self.goal_sampler_function()
            interventions_dict.update(goal_interventions)
        
        # 2. physical properties intervention (40% probability)
        if self._rng.random() < 0.4:
            physical_interventions = self._sample_physical_properties()
            interventions_dict.update(physical_interventions)
        
        # 3. visual intervention (50% probability)
        if self._rng.random() < 0.5:
            visual_interventions = self._sample_visual_properties()
            interventions_dict.update(visual_interventions)
        
        # 4. rigid pose intervention (60% probability)
        if self._rng.random() < 0.6:
            pose_interventions = self._sample_rigid_poses()
            interventions_dict.update(pose_interventions)
        
        # 5. combined multi-modal intervention (20% intervention)
        # this creates more complex combinations not seen during individual training
        if self._rng.random() < 0.2:
            multimodal_interventions = self._sample_multimodal_interventions()
            interventions_dict.update(multimodal_interventions)
        
        return interventions_dict
    
    def _sample_physical_properties(self):
        """sample physical properties with validation-specific ranges"""
        interventions_dict = dict()

        # focus on combinations not typically used in training
        target_groups = ['tool', 'stage', 'robot']
        selected_group = self._rng.choice(target_groups)

        for variable in self.task_intervention_space:
            if variable.startswith(selected_group):
                if isinstance(self.task_intervention_space[variable], dict):
                    if 'mass' in self.task_intervention_space[variable]:
                        interventions_dict[variable] = dict()
                        # sample from the upper 30% of mass range for validation
                        mass_range = self.task_intervention_space[variable]['mass']
                        mass_min = mass_range[0] + 0.7 * (mass_range[1] - mass_range[0])
                        mass_max = mass_range[1]
                        interventions_dict[variable]['mass'] = self._rng.uniform(mass_min, mass_max)
                    
                    if 'friction' in self.task_intervention_space[variable]:
                        if variable not in interventions_dict:
                            interventions_dict[variable] = dict()
                        # sample from the lower 30% of friction range for validation
                        friction_range = self.task_intervention_space[variable]['friction']
                        friction_min = friction_range[0]
                        friction_max = friction_range[0] + 0.3 * (friction_range[1] - friction_range[0])
                        interventions_dict[variable]['friction'] = self._rng.uniform(friction_min, friction_max)
                
                elif 'mass' in variable:
                    mass_range = self.task_intervention_space[variable]
                    mass_min = mass_range[0] + 0.7 * (mass_range[1] - mass_range[0])
                    mass_max = mass_range[1]
                    interventions_dict[variable] = self._rng.uniform(mass_min, mass_max)

                elif 'friction' in variable:
                    friction_range = self.task_intervention_space[variable]
                    friction_min = friction_range[0]
                    friction_max = friction_range[0] + 0.3 * (friction_range[1] - friction_range[0])
                    interventions_dict[variable] = self._rng.uniform(friction_min, friction_max)

        return interventions_dict
    
    def _sample_visual_properties(self):
        """sample visual properties with validation-specific color schemes"""
        interventions_dict = dict()

        # creating validation specific color palettes
        validation_color_palettes = [
            [0.8, 0.2, 0.2],    # reddish
            [0.2, 0.8, 0.2],    # greenish
            [0.2, 0.2, 0.8],    # blueish
            [0.8, 0.8, 0.2],    # yellowish
            [0.8, 0.2, 0.8],    # magenta-ish
        ]

        selected_palette = self._rng.choice(len(validation_color_palettes))
        base_color = validation_color_palettes[selected_palette]

        for variable in self.task_intervention_space:
            if isinstance(self.task_intervention_space[variable], dict):
                if 'color' in self.task_intervention_space[variable]:
                    interventions_dict[variable] = dict()
                    # add some noise to base color while keeping it in range
                    color_noise = self._rng.uniform(-0.1, 0.1, 3)
                    target_color = np.clip(np.array(base_color) + color_noise, 0.0, 1.0)
                    interventions_dict[variable]['color'] = target_color
            elif 'color' in variable:
                color_noise = self._rng.uniform(-0.1, 0.1, 3)
                target_color = np.clip(np.array(base_color) + color_noise, 0.0, 1.0)
                interventions_dict[variable] = target_color
        
        return interventions_dict

    def _sample_rigid_poses(self):
        """sample rigid poses with validation-specific constraints"""
        interventions_dict = dict()
        for variable in self.task_intervention_space:
            if variable.startswith('tool'):
                interventions_dict[variable] = dict()
                # position: sample from edge regions for validation
                if 'cylindrical_position' in self.task_intervention_space[variable]:
                    pos_range = self.task_intervention_space[variable]['cylindrical_position']
                    # sample from outer 25% of position range
                    if self._rng.random() < 0.5:
                        # lower edge
                        pos_min = pos_range[0]
                        pos_max = pos_range[0] + 0.25 * (pos_range[1] - pos_range[0])
                    else:
                        # upper edge
                        pos_min = pos_range[0] + 0.75 * (pos_range[1] - pos_range[0])
                        pos_max = pos_range[1]
                    interventions_dict[variable]['cylindrical_position'] = self._rng.uniform(pos_min, pos_max)
                
                # orientation: sample from challenging orientations
                if 'euler_orientation' in self.task_intervention_space[variable]:
                    ori_range = self.task_intervention_space[variable]['euler_orientation']
                    # creating challenging orientations by sampling from extremes
                    challenging_orientations = []
                    for i in range(len(ori_range[0])):
                        if self._rng.random() < 0.3:
                            # sample from extreme values
                            challenging_orientations.append(self._rng.choice([ori_range[0][i], ori_range[1][i]]))
                        else:
                            # sample normally but bias toward edges
                            if self._rng.random() < 0.5:
                                val = self._rng.uniform(ori_range[0][i], ori_range[0][i] + 0.3 * (ori_range[1][i] - ori_range[0][i]))
                            else:
                                val = self._rng.uniform(ori_range[0][i] + 0.7 * (ori_range[1][i] - ori_range[0][i]), ori_range[1][i])
                            challenging_orientations.append(val)
                    interventions_dict[variable]['euler_orientation'] = challenging_orientations
        
        return interventions_dict
    
    def _sample_multimodal_interventions(self):
        """create complex multimodal interventions not seen during training"""
        interventions_dict = dict()

        # scenario 1: heavy object with low friction and extreme color
        if self._rng.random() < 0.33:
            for variable in self.task_intervention_space:
                if variable.startswith('tool') and isinstance(self.task_intervention_space[variable], dict):
                    interventions_dict[variable] = dict()
                    if 'mass' in self.task_intervention_space[variable]:
                        mass_range = self.task_intervention_space[variable]['mass']
                        # heavy mass (top 20%)
                        interventions_dict[variable]['mass'] = self._rng.uniform(mass_range[0] + 0.8 * (mass_range[1] - mass_range[0]), mass_range[1])
                    if 'color' in self.task_intervention_space[variable]:
                        # extreme color (very dark or very bright)
                        if self._rng.random() < 0.5:
                            interventions_dict[variable]['color'] = self._rng.uniform(0.0, 0.2, 3)  # dark
                        else:
                            interventions_dict[variable]['color'] = self._rng.uniform(0.8, 1.0, 3)  # bright
                    if 'cylindrical_position' in self.task_intervention_space[variable]:
                        # edge position
                        pos_range = self.task_intervention_space[variable]['cylindrical_position']
                        interventions_dict[variable]['cylindrical_position'] = self._rng.uniform(
                            pos_range[0], pos_range[0] + 0.2 * (pos_range[1] - pos_range[0])
                        )
        
        # scenario 2: light object with high friction and unusual orientation
        elif self._rng.random() < 0.5:
            for variable in self.task_intervention_space:
                if variable.startswith('tool') and isinstance(self.task_intervention_space[variable], dict):
                    interventions_dict[variable] = dict()
                    if 'mass' in self.task_intervention_space[variable]:
                        mass_range = self.task_intervention_space[variable]['mass']
                        # light mass (bottom 20%)
                        interventions_dict[variable]['mass'] = self._rng.uniform(
                            mass_range[0], mass_range[0] + 0.2 * (mass_range[1] - mass_range[0])
                        )
                    if 'euler_orientation' in self.task_intervention_space[variable]:
                        ori_range = self.task_intervention_space[variable]['euler_orientation']
                        # extreme orientations
                        extreme_ori = []
                        for i in range(len(ori_range[0])):
                            extreme_ori.append(self._rng.choice([ori_range[0][i], ori_range[1][i]]))
                        interventions_dict[variable]['euler_orientation'] = extreme_ori
        
        return interventions_dict

    def get_params(self):
        """returns parameters for recreating the validation intervention actor """
        return {
            'validation_actor': {
                'seed': self.seed,
                'intervention_probability': self.intervention_probability
            }
        }
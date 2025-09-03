import numpy as np
from causal_world.intervention_actors.base_actor import BaseInterventionActorPolicy
import logging

class ValidationInterventionActorPolicy(BaseInterventionActorPolicy):

    def __init__(self, seed=0, intervention_probability=0.7, fixed_intervention=False, **kwargs):
        """
        here, the intervention actor creates a validation set by combining multiple
        intervention types in a deterministic, seed-based manner.

        :param seed: (int) the random seed for deterministic intervention sampling
        :param intervention_probability: (float) probability of applying each intervention type
        :param fixed_intervention: (bool) if True, use the same intervention pattern for all calls
        :param kwargs: additional parameters
        """
        super(ValidationInterventionActorPolicy, self).__init__()
        self.seed = seed
        self.intervention_probability = intervention_probability
        self.task_intervention_space = None
        self.goal_sampler_function = None
        self._rng = np.random.RandomState(seed)
        self.fixed_intervention = fixed_intervention

        # track intervention call count for seed variation
        self._call_count = 0

        # store the generated intervention for reuse if fixed_intervention is True
        self.cached_intervention = None
    
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
        # if we want fixed interventions and have already generated one, reuse it
        if self.fixed_intervention and self.cached_intervention is not None:
            return self.cached_intervention

        try:
            # only vary the seed if we're not using fixed interventions
            if not self.fixed_intervention:
                current_seed = self.seed + self._call_count
                self._rng = np.random.RandomState(current_seed)
                self._call_count += 1
            else:
                # for fixed interventions, always use the base seed
                self._rng = np.random.RandomState(self.seed)

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
            
            # applying the bounds here to ensure that all interventions are valid
            if interventions_dict:
                interventions_dict = self._check_intervention_bounds(interventions_dict)

            if not interventions_dict:
                logging.warning("Generated empty intervention dict - using fallback intervention")
                # create a simple fallback intervention
                interventions_dict = self._generate_fallback_interventions()
            
            # we cache the intervention if we're using fixed interventions
            if self.fixed_intervention:
                self.cached_intervention = interventions_dict.copy()
            
            return interventions_dict
        
        except Exception as e:
            logging.error(f"Error creating validation interventions: {e}")
            import traceback
            logging.error(traceback.format_exc())
            raise   # return empty dict or error to avoid downstream failures
    
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
                        mass_min = max(mass_range[0], mass_range[0] + 0.7 * (mass_range[1] - mass_range[0]))
                        mass_max = mass_range[1]
                        interventions_dict[variable]['mass'] = self._rng.uniform(mass_min, mass_max)
                    
                    if 'friction' in self.task_intervention_space[variable]:
                        if variable not in interventions_dict:
                            interventions_dict[variable] = dict()
                        # sample from the lower 30% of friction range for validation
                        friction_range = self.task_intervention_space[variable]['friction']
                        friction_min = friction_range[0]
                        friction_max = min(friction_range[1], friction_range[0] + 0.3 * (friction_range[1] - friction_range[0]))
                        if friction_min >= friction_max:
                            friction_max = friction_range[1]
                        interventions_dict[variable]['friction'] = self._rng.uniform(friction_min, friction_max)
                
                elif 'mass' in variable:
                    mass_range = self.task_intervention_space[variable]
                    mass_min = max(mass_range[0], mass_range[0] + 0.7 * (mass_range[1] - mass_range[0]))
                    mass_max = mass_range[1]
                    if mass_min >= mass_max:
                        mass_min = mass_range[0]
                    interventions_dict[variable] = self._rng.uniform(mass_min, mass_max)

                elif 'friction' in variable:
                    friction_range = self.task_intervention_space[variable]
                    friction_min = friction_range[0]
                    friction_max = min(friction_range[1], friction_range[0] + 0.3 * (friction_range[1] - friction_range[0]))
                    if friction_min >= friction_max:
                        friction_max = friction_range[1]
                    interventions_dict[variable] = self._rng.uniform(friction_min, friction_max)

        return interventions_dict
    
    def _sample_visual_properties(self):
        """sample visual properties with validation-specific color schemes"""
        interventions_dict = dict()

        # generating base color palette that will be used with variations
        base_color_r = np.random.uniform(0.7, 0.9)  # red component
        base_color_g = np.random.uniform(0.7, 0.9)  # green component
        base_color_b = np.random.uniform(0.7, 0.9)  # blue component
        base_color = np.array([base_color_r, base_color_g, base_color_b])

        for variable in self.task_intervention_space:
            if isinstance(self.task_intervention_space[variable], dict):
                if 'color' in self.task_intervention_space[variable]:
                    color_bounds = self.task_intervention_space[variable]['color']
                    lower_bound = color_bounds[0]
                    upper_bound = color_bounds[1]

                    interventions_dict[variable] = dict()

                    # sample within the valid bounds
                    color = lower_bound + (upper_bound - lower_bound) * base_color

                    # add some noise to base color while keeping it in range
                    noise = self._rng.uniform(-0.1, 0.1, 3) * (upper_bound - lower_bound)
                    color = np.clip(color + noise, lower_bound, upper_bound)

                    interventions_dict[variable]['color'] = color

            elif 'color' in variable:
                # now we direct the color variables (like the floor color, etc.)
                color_bounds = self.task_intervention_space[variable]
                lower_bound = color_bounds[0]
                upper_bound = color_bounds[1]

                # map the base color to the valid range
                color = lower_bound + (upper_bound - lower_bound) * base_color

                # add small noise while staying in bounds
                noise = self._rng.uniform(-0.1, 0.1, 3) * (upper_bound - lower_bound)
                color = np.clip(color + noise, lower_bound, upper_bound)

                interventions_dict[variable] = color
        
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
                        # get the actual color bounds
                        color_bounds = self.task_intervention_space[variable]['color']
                        lower_bound = color_bounds[0]
                        upper_bound = color_bounds[1]
                        
                        # close to bounds but still valid
                        if self._rng.random() < 0.5:
                            lower_margin = 0.1 * (upper_bound - lower_bound)
                            interventions_dict[variable]['color'] = self._rng.uniform(lower_bound, lower_bound + lower_margin, 3)  # dark
                        else:
                            upper_margin = 0.1 * (upper_bound - lower_bound)
                            interventions_dict[variable]['color'] = self._rng.uniform(upper_bound - upper_margin, upper_bound, 3)  # bright
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

    def _check_intervention_bounds(self, intervention_dict):
        """check and fix any interventions that might violate bounds"""
        fixed_dict = {}

        # a deep copy to avoid modifying the original
        for key, value in intervention_dict.items():
            if isinstance(value, dict):
                fixed_dict[key] = {}
                for inner_key, inner_value in value.items():
                    if inner_key == 'color' and isinstance(inner_value, np.ndarray):
                        # ensure colors respect the bounds
                        if key in self.task_intervention_space and inner_key in self.task_intervention_space[key]:
                            color_bounds = self.task_intervention_space[key][inner_key]
                            lower_bound = color_bounds[0]
                            upper_bound = color_bounds[1]
                            # get the colors to respect their specific bounds
                            fixed_dict[key][inner_key] = np.clip(inner_value, lower_bound, upper_bound)
                        else:
                            # this is a fallback in case we can't find specific bounds
                            fixed_dict[key][inner_key] = np.clip(inner_value, 0.5, 1.0)
                    else:
                        # for other properties, keep as is
                        fixed_dict[key][inner_key] = inner_value
            elif 'color' in key and isinstance(value, np.ndarray):
                # direct color properties
                if key in self.task_intervention_space:
                    color_bounds = self.task_intervention_space[key]
                    lower_bound = color_bounds[0]
                    upper_bound = color_bounds[1]
                    fixed_dict[key] = np.clip(value, lower_bound, upper_bound)
                else:
                    # here we fallback to the default bounds
                    fixed_dict[key] = np.clip(value, 0.5, 1.0)
            else:
                fixed_dict[key] = value
        
        return fixed_dict
    
    def _generate_fallback_interventions(self):
        """
        generates a simple, guaranteed intervention when no other interventions are seleted.
        this ensures that the validation env is always modified.
        """
        return self._sample_visual_properties()
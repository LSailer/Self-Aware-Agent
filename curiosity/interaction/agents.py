'''
Agents have access to the graph and make action choices each turn.
'''
import numpy as np
import random
import copy


class Agent(object):
    '''
    Abstract agent class

    '''
    def __init__(self,
            compute_settings,
            stored_params,
            **kwargs):
        raise NotImplementedError()


    def act(self,
            sess,
            history):
        raise NotImplementedError()


class TelekineticMagicianAgent(Agent):
    '''
    The baby.
    '''
    def __init__(self,
            compute_settings,
            stored_params,
            **kwargs):
        # Initialization logic
        super().__init__(compute_settings, stored_params, **kwargs)
        self.compute_settings = compute_settings
        self.stored_params = stored_params
        # Additional initializations specific to this agent

    def act(self,
            sess,
            history):
        '''
        Determine the next action for the agent.
        
        Args:
            sess: TensorFlow session.
            history: Historical interaction data.

        Returns:
            Action choice.
        '''
        # TODO: Implement logic for selecting the next action. Passe die act-Methode so an, dass sie Entscheidungen auch basierend auf der Kommunikation mit anderen Agenten trifft.
        # Implement logic for selecting the next action
        action = None
        return action

    def update(self,
               reward,
               new_state):
        '''
        Update the agent's internal state or learning parameters.

        Args:
            reward: Reward received from the environment.
            new_state: The new state of the environment after the action.
        '''
        # Implement update logic, e.g., updating internal models or strategies
        pass
    def communicate(self, other_agent):
        # TODO: Implement communication logic Informationen mit anderen Agenten auszutauschen
        shared_info = {'position': self.position, 'goal': self.current_goal}
        return shared_info
    
#TODO: Erstelle eine neue Klasse MultiAgentEnvironment, die die Interaktionen zwischen mehreren Agenten koordiniert.
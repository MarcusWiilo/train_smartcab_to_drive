import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import matplotlib.pyplot as plt

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.state = {}
        self.epsilon =.2
        self.epsilon_annealing_rate = .01
        self.episilon_reset_trials = 200
        self.alpha = 0.6
        self.gama = 0.4
        self.q_table = {}
        self.valid_actions = self.env.valid_actions
        self.total_wins = 0
        self.trial_infractions = 0
        self.infractions_record = []
        self.trial_count = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.infractions_record.append(self.trial_infractions)
        self.trial_infractions = 0
        self.trial_count += 1
        if self.trial_count < self.episilon_reset_trials:
            self.epsilon = .05

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = self.set_current_state(inputs)

        # TODO: Select action according to your policy
        action = self.choose_action(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def set_current_state(self, inputs):
        return {"light": inputs["light"], "oncoming": inputs["oncoming"], "left": inputs["left"], "next_waypoint": self.next_waypoint
        }

    def choose_action(self, state):
        if random.random() < self.epsilon:
            self.epsilon -= self.epsilon_annealing_rate
            return random.choice(self.valid_actions)

        opt_action = self.valid_actions[0]
        opt_value = 0

        for action in self.valid_actions:
            cur_value = self.q_value(state, action)
            if cur_value > opt_value:
                opt_action = action
                opt_value = cur_value
            elif cur_value == opt_value:
                opt_action = random.choice([opt_action, action])
        return opt_action

    def q_value(self, state, action):
        q_key = self.q_key(state, action)
        if q_key in self.q_table:
            return self.q_table[q_key]
        return 0

    def q_key(self, state, action):
        return "{}.{}.{}.{}.{}".format(state["light"], state["next_waypoint"], state["oncoming"], state["left"], action)

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

if __name__ == '__main__':
    run()
